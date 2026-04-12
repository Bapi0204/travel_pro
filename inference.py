import os
import json
import re
import sys
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

# Configuration - Prioritize Env Vars for Evaluation
# Configuration - Prioritize Env Vars for Evaluation
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "travel_pro")
LEVEL = os.getenv("LEVEL", "ALL")  # Default to running ALL levels (1, 2, 3)
MAX_STEPS = 8

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert travel booking agent interacting with a travel environment.
    Your goal is to fulfill the user's travel goal within budget and constraints.
    
    ACTION SCHEMAS (STRICT JSON ONLY):
    1. Search: {"action_type": "Search", "parameters": {"query": "<city or flight search>"}}
    2. Book: {"action_type": "Book", "parameters": {"item_id": 1, "item_type": "flight"|"hotel"}}
    3. Finalize: {"action_type": "Finalize", "parameters": {}}
    
    IMPORTANT RULES:
    - NEVER use vague queries like "current destination". Use REAL city names (e.g., Paris, Mumbai, London).
    - You MUST "Book" at least ONE flight AND one hotel before calling "Finalize".
    - If a search returns 0 results, CHANGE your query strategy. Try different cities or routes.
    - Level 2/3 involve strikes or price changes. If you see "Price expired", you MUST re-search.
    - If you are stuck, analyze the "Error Logs" to understand what went wrong.
    
    Return your reasoning and action as a JSON object.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Format: [STEP] followed by exactly TWO spaces
    print(f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Format: [END] followed by exactly THREE spaces
    print(f"[END]   success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def get_model_response(client: OpenAI, goal: str, obs: Any, history: List[str]) -> Dict[str, Any]:
    obs_data = obs.model_dump()
    last_error = obs_data['error_log'][-1] if obs_data['error_log'] else "None"
    
    user_prompt = textwrap.dedent(f"""
        [GOAL]
        {goal}

        [CURRENT STATE]
        Balance: ${obs_data['balance']:.2f}
        Itinerary: {obs_data['itinerary']}
        Last Action Result/Error: {last_error}
        
        [AVAILABLE OPTIONS (First 10)]
        {chr(10).join(obs_data['available_options'][:10]) if obs_data['available_options'] else "None"}
        
        [HISTORY]
        {chr(10).join(history[-3:]) if history else "No history yet."}
        
        Analyze the Last Action Result. If it failed, adjust your strategy.
        Respond with your 'reasoning' and 'action' in JSON.
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
        print(f"DEBUG: LLM Inference Error: {e}", file=sys.stderr)
        # Smarter Fallback: Try to search for the goal destination if model fails
        fallback_query = "Paris" # Default fallback
        dest_match = re.search(r"to ([A-Za-z\s]+)", goal)
        if dest_match:
            fallback_query = dest_match.group(1).strip()
            
        return {
            "reasoning": f"Critical Error: {e}. Attempting recovery search for {fallback_query}.",
            "action": {"action_type": "Search", "parameters": {"query": fallback_query}}
        }

def run_task(level: int):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    level_names = {1: "HAPPY_PATH", 2: "ADVERSARIAL", 3: "CHAOS"}
    task_name = f"level_{level}_{level_names.get(level, 'UNKNOWN')}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    env = TravelEnv()
    obs = env.reset(level=level)
    
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
                action = TravelAction(type="search", query=params.get("query", ""))
            elif act_type == "Book":
                action = TravelAction(type="book", item_id=params.get("item_id"), item_type=params.get("item_type"))
            else:
                action = TravelAction(type="finalize")
        except Exception as e:
            reasoning = f"PARSING ERROR: {e}"
            action = TravelAction(action=Finalize())
        
        obs = env.step(action)
        reward = obs.reward or 0.0
        done = obs.done
        rewards.append(reward)
        
        # Log this step in standardized format
        log_step(
            step=step, 
            action=json.dumps(action_data), 
            reward=reward, 
            done=done, 
            error=obs.error_log[-1] if obs.error_log else None
        )
        
        history.append(f"Step {step}: {reasoning} -> Action: {act_type}")
        
        if done:
            break

    # Comprehensive Grading
    state = env.state
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
        rewards=rewards
    )

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: API_KEY/HF_TOKEN not found in environment.")
    else:
        if LEVEL.upper() == "ALL":
            levels_to_run = [1, 2, 3]
        else:
            try:
                levels_to_run = [int(LEVEL)]
            except ValueError:
                print(f"ERROR: Invalid LEVEL '{LEVEL}'. Defaulting to 1.")
                levels_to_run = [1]

        for level in levels_to_run:
            try:
                run_task(level)
            except Exception as e:
                print(f"Failed to run level {level}: {e}", flush=True)
