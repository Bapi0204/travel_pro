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

class StateTracker:
    def __init__(self):
        self.has_flight = False
        self.has_hotel = False
        self.last_search_flights = 0
        self.last_search_hotels = 0
        self.itinerary = []
        self.last_error = ""
        self.failed_bookings = set() # (item_id, item_type)
        
    def update(self, obs):
        self.itinerary = obs.itinerary
        self.has_flight = any("Flight" in item for item in self.itinerary)
        self.has_hotel = any("Hotel" in item for item in self.itinerary)
        self.last_error = obs.error_log[-1] if obs.error_log else ""
        
        # Parse last search results from error log if available
        if self.last_error and "found" in self.last_error and "flights" in self.last_error:
            try:
                match = re.search(r"found (\d+) flights and (\d+) hotels", self.last_error)
                if match:
                    self.last_search_flights = int(match.group(1))
                    self.last_search_hotels = int(match.group(2))
            except Exception: pass

QUERY_GEN_PROMPT = textwrap.dedent("""
    You are a query generator for a travel agent.
    Goal: {goal}
    History: {history}
    Last Error: {error}
    
    Task: Generate a specific Search query (e.g., 'flights to Paris' or 'hotels in London').
    If previous searches failed, try a different city or route.
    Return JSON: {{"reasoning": "...", "query": "..."}}
""").strip()

CHOICE_PROMPT = textwrap.dedent("""
    You are a selection agent for a travel agent.
    Goal: {goal}
    Available Options: {options}
    Itinerary: {itinerary}
    
    Task: Pick the best item_id for a {item_type}.
    Your response MUST include "item_id" as a raw INTEGER (no strings, e.g. 9 instead of "Flight 9").
    Ensure it meets constraints (e.g., rating, direct flight).
    Return JSON: {{"reasoning": "...", "item_id": 123}}
""").strip()

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

def get_choice(client: OpenAI, goal: str, options: List[str], itinerary: List[str], item_type: str, budget: float) -> Dict[str, Any]:
    try:
        prompt = CHOICE_PROMPT.format(goal=goal, options=options[:10], itinerary=itinerary, item_type=item_type)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt + f"\nCurrent Balance: ${budget:.2f}. Pick an affordable option."}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        if "```json" in content: content = content.split("```json")[1].split("```")[0]
        data = json.loads(content)
        
        # Robustly extract item_id as integer
        raw_id = data.get("item_id")
        if isinstance(raw_id, str):
            id_match = re.search(r"(\d+)", raw_id)
            if id_match: data["item_id"] = int(id_match.group(1))
            else: data["item_id"] = 1
        elif raw_id is None:
            data["item_id"] = 1
        return data
    except Exception as e:
        print(f"DEBUG: Choice Error: {e}", file=sys.stderr)
        return {"item_id": 1}

def get_query(client: OpenAI, goal: str, history: List[str], error: str) -> Dict[str, Any]:
    try:
        prompt = QUERY_GEN_PROMPT.format(goal=goal, history=history[-3:], error=error)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        content = completion.choices[0].message.content
        if "```json" in content: content = content.split("```json")[1].split("```")[0]
        return json.loads(content)
    except Exception as e:
        print(f"DEBUG: Query Error: {e}", file=sys.stderr)
        fallback = "Paris"
        match = re.search(r"to ([A-Za-z\s]+)", goal)
        if match: fallback = match.group(1).strip()
        return {"query": fallback}

def run_task(level: int):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    level_names = {1: "HAPPY_PATH", 2: "ADVERSARIAL", 3: "CHAOS"}
    task_name = f"level_{level}_{level_names.get(level, 'UNKNOWN')}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    env = TravelEnv()
    obs = env.reset(level=level)
    tracker = StateTracker()
    tracker.update(obs)
    
    rewards = []
    history = []
    
    for step in range(1, MAX_STEPS + 1):
        # State machine logic
        if "Booking failed" in tracker.last_error:
            # Pivot back to search for different options if booking fails
            query_res = get_query(client, str(obs.current_goal), history, f"FAILURE: {tracker.last_error}. Try to find cheaper options.")
            action = TravelAction(type="search", query=query_res.get("query", "Paris"))
            reasoning = f"Booking failed, re-searching: {query_res.get('reasoning')}"
            tracker.last_search_flights = 0 # Force re-search state
            tracker.last_search_hotels = 0
        elif not tracker.has_flight and tracker.last_search_flights > 0:
            choice = get_choice(client, str(obs.current_goal), obs.available_options, tracker.itinerary, "flight", obs.balance)
            action = TravelAction(type="book", item_id=choice.get("item_id", 1), item_type="flight")
            reasoning = choice.get("reasoning", "Booking flight.")
        elif tracker.has_flight and not tracker.has_hotel and tracker.last_search_hotels > 0:
            choice = get_choice(client, str(obs.current_goal), obs.available_options, tracker.itinerary, "hotel", obs.balance)
            action = TravelAction(type="book", item_id=choice.get("item_id", 1), item_type="hotel")
            reasoning = choice.get("reasoning", "Booking hotel.")
        elif tracker.has_flight and tracker.has_hotel:
            action = TravelAction(type="finalize")
            reasoning = "Itinerary complete. Finalizing."
        else:
            query_res = get_query(client, str(obs.current_goal), history, tracker.last_error if tracker.last_error else "None")
            action = TravelAction(type="search", query=query_res.get("query", "Paris"))
            reasoning = query_res.get("reasoning", "Searching for options.")
        
        # Execute
        obs = env.step(action)
        tracker.update(obs)
        
        reward = obs.reward or 0.0
        done = obs.done
        rewards.append(reward)
        
        # Log
        log_step(
            step=step, 
            action=action.model_dump_json(exclude_none=True), 
            reward=reward, 
            done=done, 
            error=obs.error_log[-1] if obs.error_log else "null"
        )
        
        history.append(f"Step {step}: {reasoning} -> {action.type}")
        if done: break

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
