import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import textwrap
from typing import List, Optional
from openai import OpenAI
from travel_pro.server.env import TravelEnv
from travel_pro.models import TravelAction, Search, Book, Finalize

# Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")

# Auto-detect routing if not explicitly set
if not API_BASE_URL:
    if str(API_KEY).startswith("sk-"):
        API_BASE_URL = "https://api.openai.com/v1"
        MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"
    else:
        API_BASE_URL = "https://router.huggingface.co/v1"
        MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
else:
    MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "travel_pro"
MAX_STEPS = 15

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert travel booking agent. Your goal is to fulfill the user's travel goal within budget and constraints.
    Return your next action as a JSON object with a "reasoning" field and a "parameters" field:
    
    ACTION SCHEMAS:
    1. {"reasoning": "...", "action_type": "Search", "parameters": {"query": "..."}}
    2. {"reasoning": "...", "action_type": "Book", "parameters": {"item_id": 1, "item_type": "flight"}}
    3. {"reasoning": "...", "action_type": "Book", "parameters": {"item_id": 1, "item_type": "hotel"}}
    4. {"reasoning": "...", "action_type": "Finalize", "parameters": {}}
    
    IMPORTANT RULES:
    - You MUST use the exact integer "item_id" from the available options (e.g., if the option says "Flight 4", use "item_id": 4).
    - You MUST specify "item_type" as either "flight" or "hotel".
    - You MUST "Book" at least ONE flight AND one hotel before calling "Finalize".
    - If your current itinerary is empty, START with a "Search".
    
    Return ONLY the raw JSON object.
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, goal: str, obs: str, history: List[str]) -> str:
    user_prompt = f"Goal: {goal}\nObservation: {obs}\nHistory: {history}\nNext action (JSON):"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        return json.dumps({"action_type": "Finalize", "parameters": {}})

def parse_action(content: str) -> TravelAction:
    try:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            act_type = data.get("action_type")
            params = data.get("parameters", {})
            if act_type == "Search":
                return TravelAction(action=Search(query=params.get("query", "")))
            elif act_type == "Book":
                return TravelAction(action=Book(item_id=params.get("item_id"), item_type=params.get("item_type")))
            elif act_type == "Finalize":
                return TravelAction(action=Finalize())
    except Exception as e:
        print(f"--- [Parser Error] {e} Content: {content} ---")
        pass
    return TravelAction(action=Finalize())

def run_task(level: int):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = TravelEnv()
    obs = env.reset(level=level)
    
    task_name = f"level_{level}"
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    history = []
    
    for step in range(1, MAX_STEPS + 1):
        action_str = get_model_action(client, str(obs.current_goal), str(obs), history)
        action = parse_action(action_str)
        
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        steps_taken = step
        
        log_step(step=step, action=action_str.replace("\n", " "), reward=reward, done=done, error=None)
        history.append(f"Step {step}: {action_str} -> {reward}")
        
        if done:
            break
            
    # Calculate score using EfficiencyGrader logic for now
    from travel_pro.evaluator import EfficiencyGrader
    score = EfficiencyGrader.grade(env.state())
    success = (score > 0 and len(obs.itinerary) >= 2 and not any("Violation" in log for log in obs.error_log))
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    for level in [1, 2, 3]:
        run_task(level)
