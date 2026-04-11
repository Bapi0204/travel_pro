import os
import json
import re
import textwrap
from typing import List, Optional, Dict, Any
from openai import OpenAI
from pydantic import BaseModel, Field

# Load environment
from dotenv import load_dotenv
load_dotenv()

from travel_pro.server.env import TravelEnv
from travel_pro.models import TravelAction, Search, Book, Finalize
from travel_pro.evaluator import EfficiencyGrader, BudgetOptimizationGrader, ConstraintGrader

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"
BENCHMARK = "travel_pro"
MAX_STEPS = 15

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert travel booking agent. Your goal is to fulfill the user's travel goal within budget and constraints.
    
    CAPABILITIES & CHALLENGES:
    - Level 1 (Happy Path): Standard booking with high availability.
    - Level 2 (Adversarial): Dynamic events like HOTEL_STRIKE (limiting hotels) or FLIGHT_CRUNCH (high prices). Strict rating and direct-flight constraints.
    - Level 3 (Chaos): Extreme price volatility. Prices change every step. If you see "Price expired", you MUST re-search to get current prices.
    
    INSTRUCTIONS:
    1. Analyze the current Goal and Observation.
    2. Provide a detailed reasoning for your next step.
    3. Return your next action as a JSON object within your response.
    
    ACTION SCHEMAS:
    - {"action_type": "Search", "parameters": {"query": "..."}}
    - {"action_type": "Book", "parameters": {"item_id": 1, "item_type": "flight"}}
    - {"action_type": "Book", "parameters": {"item_id": 1, "item_type": "hotel"}}
    - {"action_type": "Finalize", "parameters": {}}
    
    IMPORTANT RULES:
    - You MUST "Book" at least ONE flight AND one hotel before calling "Finalize".
    - If you encounter a constraint violation or price expiry, adjust your strategy.
    - If your itinerary is empty, always start with a "Search".
    
    Your output MUST be a JSON object with two fields: "reasoning" and "action".
    Example:
    {
        "reasoning": "Since I need to go to Paris and have no results, I'll search for flights.",
        "action": {"action_type": "Search", "parameters": {"query": "Flights to Paris"}}
    }
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"\n[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, reasoning: str, action: str, reward: float, done: bool, error: Optional[str], balance: float) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Demo-style verbose reasoning in the action field
    verbose_action = f"{reasoning} -> EXECUTING {action}"
    print(f"[STEP] step={step} action={verbose_action} reward={reward:.2f} done={done_val} error={error_val} balance={balance:.2f}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float], breakdown: Dict[str, float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    breakdown_str = " | ".join([f"{k}: {v:.2f}" for k, v in breakdown.items()])
    print(f"[END] success={str(success).lower()} steps={steps} final_score={score:.3f} rewards=[{rewards_str}]", flush=True)
    print(f"      Score Breakdown: {breakdown_str}", flush=True)

def get_model_response(client: OpenAI, goal: str, obs: Any, history: List[str]) -> Dict[str, Any]:
    # Constructing a rich prompt with environment state
    obs_data = obs.model_dump()
    user_prompt = textwrap.dedent(f"""
        User Goal: {goal}
        Current Balance: ${obs_data['balance']:.2f}
        Current Itinerary: {obs_data['itinerary']}
        Error Logs: {obs_data['error_log']}
        Available Options: {obs_data['available_options'][:10]}
        
        Action History:
        {chr(10).join(history[-5:]) if history else "No history yet."}
        
        Return your reasoning and next action in JSON format.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        return {
            "reasoning": f"Error communicating with model: {e}",
            "action": {"action_type": "Finalize", "parameters": {}}
        }

def run_task(level: int):
    client = OpenAI(api_key=OPENAI_API_KEY)
    env = TravelEnv()
    obs = env.reset(level=level)
    
    level_names = {1: "HAPPY_PATH", 2: "ADVERSARIAL", 3: "CHAOS"}
    task_name = f"level_{level}_{level_names.get(level, 'UNKNOWN')}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    history = []
    
    for step in range(1, MAX_STEPS + 1):
        response = get_model_response(client, str(obs.current_goal), obs, history)
        reasoning = response.get("reasoning", "No reasoning provided.")
        action_data = response.get("action", {})
        
        # Convert to TravelAction model
        act_type = action_data.get("action_type")
        params = action_data.get("parameters", {})
        
        try:
            if act_type == "Search":
                action = TravelAction(action=Search(query=params.get("query", "")))
            elif act_type == "Book":
                action = TravelAction(action=Book(item_id=params.get("item_id"), item_type=params.get("item_type")))
            else:
                action = TravelAction(action=Finalize())
        except Exception as e:
            reasoning = f"PARSING ERROR: {e}"
            action = TravelAction(action=Finalize())
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        
        # Log this step in demo style
        log_step(
            step=step, 
            reasoning=reasoning, 
            action=json.dumps(action_data), 
            reward=reward, 
            done=done, 
            error=obs.error_log[-1] if obs.error_log else None,
            balance=obs.balance
        )
        
        history.append(f"Step {step}: {reasoning} -> Action: {act_type}")
        
        if done:
            break

    # Comprehensive Grading
    state = env.state()
    # Add info for graders that look at info dict
    state.info = {
        "done": obs.done,
        "itinerary_length": len(obs.itinerary),
        "error_log": obs.error_log
    }
    
    e_score = EfficiencyGrader.grade(state)
    b_score = BudgetOptimizationGrader.grade(state)
    c_score = ConstraintGrader.grade(state)
    
    # Final success criteria
    is_success = obs.done and len(obs.itinerary) >= 2 and not any("Violation" in log for log in obs.error_log)
    
    # Overall score (weighted): If failed, score is heavily reduced
    final_score = (e_score * 0.2 + b_score * 0.4 + c_score * 0.4) if is_success else (e_score * 0.1)
    
    log_end(
        success=is_success, 
        steps=len(rewards), 
        score=final_score, 
        rewards=rewards,
        breakdown={"Efficiency": e_score, "Budget": b_score, "Constraints": c_score}
    )

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment.")
    else:
        for level in [1, 2, 3]:
            try:
                run_task(level)
            except Exception as e:
                print(f"Failed to run level {level}: {e}")
