from typing import Dict, Any
from openenv.core.env_server.types import State

class EfficiencyGrader:
    """Grades agent efficiency based on step count."""
    @staticmethod
    def grade(state: Any) -> float:
        # Assuming we have access to common attributes via state or env_instance
        # For OpenEnv, graders often receive the environment state.
        # We'll follow a standard 0-1 range.
        step_count = getattr(state, "step_count", 0)
        max_steps = 15 # baseline max
        if step_count == 0: return 0.0
        score = max(0.0, 1.0 - (step_count / max_steps))
        return score

class BudgetOptimizationGrader:
    """Grades how well the agent optimized the budget."""
    @staticmethod
    def grade(state: Any) -> float:
        info = getattr(state, "info", {})
        if not info.get("done", False): return 0.0
        
        # If successfully finished, score is based on itinerary length and balance
        itinerary_len = info.get("itinerary_length", 0)
        if itinerary_len < 2: return 0.0
        
        # Placeholder for budget optimization: 1.0 if completed
        return 1.0

class ConstraintGrader:
    """Grades adherence to destination and rating constraints."""
    @staticmethod
    def grade(state: Any) -> float:
        info = getattr(state, "info", {})
        error_log = info.get("error_log", [])
        
        # Penalty for each constraint violation
        violations = sum(1 for log in error_log if "Violation" in log)
        score = max(0.0, 1.0 - (violations * 0.5))
        
        # Must have completed at least some parts of the trip
        if info.get("itinerary_length", 0) == 0: return 0.0
        
        return score
