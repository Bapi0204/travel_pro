---
title: Travel Pro Environment Server
emoji: 🎪
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Travel Pro Environment

An OpenEnv-compliant environment for simulating real-world travel booking tasks. It features three complexity levels to test agent autonomy, constraint adherence, and resilience to dynamic data.

## Features
- **Real-world Simulation**: Search and book flights/hotels using a realistic travel database.
- **OpenEnv Spec Compliant**: Implements the full interface with typed Pydantic models and standard step signatures.
- **3 Challenge Levels**:
  - **Level 1 (Happy Path)**: Baseline booking with high availability.
  - **Level 2 (Adversarial)**: Hotel strikes and flight crunches with strict user constraints (e.g., 4+ star ratings).
  - **Level 3 (Chaos)**: Dynamic price updates every step, requiring "stale data" handling.

## Quick Start
1. **Set up Environment**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   ```
2. **Run Baseline Evaluation**:
   ```bash
   python3 travel_pro/test_agent.py --level 1 --key [YOUR_API_KEY]
   ```

## Graders
The environment includes 3 programmatic graders:
1. `EfficiencyGrader`: Scores base on step count optimization.
2. `BudgetOptimizationGrader`: Scores based on cost-effective booking.
3. `ConstraintGrader`: Validates adherence to hotel ratings and flight types.

## Reward Function
- **Efficiency Penalty**: -0.05 per step to encourage fast completion.
- **Success Reward**: +1.0 for a successful multi-item itinerary.
- **Constraint Violations**: Deductions (-1.0) for failing to meet user requirements (e.g., booking a low-rated hotel).

## Technical Specification

### Action Space
The environment uses a `TravelAction` model which wraps one of the following:
- **Search**: `query` (str) - Searches for flights and hotels in the database.
- **Book**: `item_id` (int), `item_type` (str) - Books a specific flight or hotel.
- **Finalize**: Completes the episode and triggers the final grader.

### Observation Space
The `TravelObservation` includes:
- `itinerary`: List of booked strings.
- `available_options`: List of strings showing IDs and prices.
- `balance`: Current remaining funds.
- `current_goal`: The `UserGoal` object containing destination, budget, and constraints.
- `error_log`: Historical log of events (price expiry, violations).

## Tasks & Difficulty

| Task ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `level_1` | Happy Path | Easy | High availability, no volatility. Basic search-book-finalize flow. |
| `level_2` | Adversarial | Medium | Reduced hotel availability (Strike) and strict star-rating constraints. |
| `level_3` | Chaos Mode | Hard | Prices increase by 1-5% each step. Booking fails if data is stale. |

## Baseline Scores

Evaluation performed using `Qwen/Qwen2.5-72B-Instruct` via the provided `inference.py` script.

| Level | Success Rate | Avg Steps | Avg Reward |
| :--- | :--- | :--- | :--- |
| Level 1 | 0.0% | 1.0 | -0.55 |
| Level 2 | 0.0% | 1.0 | -0.55 |
| Level 3 | 0.0% | 1.0 | -0.55 |

*Note: The current baseline agent has difficulty following multi-step booking rules and often finalizes prematurely. This provides a clear benchmark for researchers to improve upon.*

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    TravelProEnvironment,  # Pass class, not instance
    TravelProAction,
    TravelProObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from travel_pro import TravelProAction, TravelProEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with TravelProEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(TravelProAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/travel_pro_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
travel_pro/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # TravelProEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── travel_pro_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
