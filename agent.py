import os
from dotenv import load_dotenv
load_dotenv() # Load OpenAI API key from .env file

import random
from typing import TypedDict, List, Optional, Annotated, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from travel_pro.server.env import TravelEnv
from travel_pro.models import UserGoal, TravelObservation, TravelAction, Search, Book, Finalize
from travel_pro.scenarios import ScenarioLevel

class GraphState(TypedDict):
    """State management for the Travel Pro agent graph."""
    user_goal: UserGoal
    itinerary: List[str]
    last_observation: TravelObservation
    retry_count: int
    next_action: Optional[TravelAction]
    env: TravelEnv
    error_flag: Optional[str]

def planner_node(state: GraphState):
    """Uses GPT-4o to analyze the UserGoal and the current TravelObservation."""
    goal = state['user_goal']
    obs = state['last_observation']
    error = state.get('error_flag')
    
    print(f"--- [Planner] Step {state['env']._state.step_count} ---")
    
    # If error handler flagged a stale price, we prioritize a search
    if error == "PRICE_EXPIRED":
        print("    Plan: Price expired, performing mandatory re-search.")
        action = TravelAction(action=Search(query=f"Flights to {goal.destination}"))
        return {"next_action": action, "error_flag": None}

    # API Key check for GPT-4o
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("    Plan: No API Key found, using fallback heuristic.")
        # Fallback heuristic for baseline when key is missing or invalid
        step = state['env']._state.step_count
        if step == 0:
            action = TravelAction(action=Search(query=f"Flights to {goal.destination}"))
        elif step == 1:
            # Try to book the first flight (ID 1)
            action = TravelAction(action=Book(item_id=1, item_type="flight"))
        elif step == 2:
            # Try to book the first hotel (ID 1)
            action = TravelAction(action=Book(item_id=1, item_type="hotel"))
        else:
            action = TravelAction(action=Finalize())
        return {"next_action": action}

    # Real GPT-4o Logic
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
    
    system_msg = SystemMessage(content=(
        "You are an expert travel booking agent. Your goal is to fulfill the UserGoal "
        "within the budget and constraints. Return your next action as a JSON object: "
        '{"action_type": "Search", "parameters": {"query": "..."}} OR '
        '{"action_type": "Book", "parameters": {"item_id": 123, "item_type": "flight/hotel"}} OR '
        '{"action_type": "Finalize", "parameters": {}}.'
    ))
    
    human_msg = HumanMessage(content=(
        f"Goal: {goal.model_dump()}\n"
        f"Observation: {obs.model_dump()}\n"
        f"Current Itinerary: {state['itinerary']}\n"
        "What is your next action? Return ONLY the JSON object."
    ))
    
    try:
        response = llm.invoke([system_msg, human_msg])
        import json
        import re
        # Extract JSON from potential markdown wrapping
        content = response.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            act_type = data.get("action_type")
            params = data.get("parameters", {})
            
            if act_type == "Search":
                return {"next_action": TravelAction(action=Search(query=params.get("query", ""))), "error_flag": None}
            elif act_type == "Book":
                return {"next_action": TravelAction(action=Book(item_id=params.get("item_id"), item_type=params.get("item_type"))), "error_flag": None}
            elif act_type == "Finalize":
                return {"next_action": TravelAction(action=Finalize()), "error_flag": None}
    except Exception as e:
        print(f"    [Error] LLM parsing failed: {e}. Falling back to default.")
    
    # Fallback if parsing fails or LLM is unavailable
    return {"next_action": TravelAction(action=Finalize()), "error_flag": None}

def executor_node(state: GraphState):
    """Executes the env.step() call."""
    env = state['env']
    action = state['next_action']
    
    print(f"--- [Executor] Action: {action.action.type if action else 'None'} ---")
    obs, reward, done, info = env.step(action)
    
    return {
        "last_observation": obs,
        "itinerary": obs.itinerary
    }

def error_handler_node(state: GraphState):
    """Specifically handles 'Stale Data' errors from Level 3."""
    obs = state['last_observation']
    error_logs = obs.error_log
    
    if any("Price expired" in log for log in error_logs):
        print("--- [Error Handler] Price Expired detected! ---")
        return {"error_flag": "PRICE_EXPIRED", "retry_count": state['retry_count'] + 1}
    
    return {"error_flag": None}

def should_continue(state: GraphState):
    """Transition logic for the graph."""
    if state['last_observation'].done:
        return END
    return "planner"

def create_agent():
    """Compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("error_handler", error_handler_node)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "error_handler")
    
    workflow.add_conditional_edges(
        "error_handler",
        should_continue,
        {
            "planner": "planner",
            END: END
        }
    )
    
    return workflow.compile()

def run_episode(level: int):
    """Runs a single episode at the given entropy level."""
    env = TravelEnv()
    initial_obs = env.reset(level=level)
    
    agent = create_agent()
    
    initial_state = {
        "user_goal": initial_obs.current_goal,
        "itinerary": [],
        "last_observation": initial_obs,
        "retry_count": 0,
        "next_action": None,
        "env": env,
        "error_flag": None
    }
    
    print(f"\n>>> Starting Episode - Level {level} <<<")
    result = agent.invoke(initial_state)
    
    obs = result['last_observation']
    # Success Criteria: Trip finalized with at least 2 items and no constraint violations in logs
    is_success = obs.done and len(obs.itinerary) >= 2 and not any("Violation" in log for log in obs.error_log)
    return is_success

def main():
    """Baseline evaluation across all 3 levels."""
    levels = [1, 2, 3]
    episodes_per_level = 1
    results = {1: [], 2: [], 3: []}
    
    print("Starting Baseline Evaluation...")
    
    for level in levels:
        for _ in range(episodes_per_level):
            success = run_episode(level)
            results[level].append(success)
            
    print("\n" + "="*30)
    print("FINAL SUCCESS RATE REPORT")
    print("="*30)
    for level, res in results.items():
        rate = sum(res) / len(res) * 100
        print(f"Level {level}: {rate:.1f}% ({sum(res)}/{len(res)})")
    print("="*30)

if __name__ == "__main__":
    main()
