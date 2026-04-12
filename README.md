---
title: Travel Pro Environment Server
emoji: 🎪
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
---

# 🌍 Travel Pro: Autonomous Booking Environment

Travel Pro is a high-fidelity environment built on the **OpenEnv** framework, designed to evaluate AI agents' ability to plan, search, and book travel in non-deterministic conditions.

## 🚀 Environment Overview

The environment simulates a travel booking API with a SQLite backend. Unlike static benchmarks, Travel Pro introduces **entropy levels** that test resilience to real-world friction.

### Challenge Levels
1.  **Level 1 (Happy Path)**: High budget, stable prices, and high availability. Testing basic tool-calling.
2.  **Level 2 (Adversarial)**: Injects **HOTEL_STRIKE** or **FLIGHT_CRUNCH**. Strict constraints: 4+ star hotels and direct flights.
3.  **Level 3 (Chaos Mode)**: Extreme **Price Volatility**. Prices change every step. Requires re-searching to handle `Price Expired`.

---

## 🧠 Core Architecture: Hybrid Robust Agent

We have transitioned from a simple LLM loop to a **Hybrid Robust Agent** architecture.

- **State Machine**: Enforces a strict `Search -> Book -> Finalize` sequence in `inference.py`.
- **Intelligent LLM Selection**: Uses **Qwen2.5-72B-Instruct** for specialized sub-tasks:
    - **Query Generation**: Formulates specific searches based on fails/errors.
    - **Item Selection**: Picks the best available option while respecting rating and budget constraints.
- **Error Awareness**: Automatically pivots back to a `Search` state if a booking fails (e.g., due to unexpected price changes or insufficient funds).

---

## 🛠️ Action & Observation Space

### Actions
The agent interacts via a `TravelAction` union:
- **`Search(query: str)`**: Intelligent filtering for destinations (e.g., "Paris", "London"). Maps natural language to DB codes.
- **`Book(item_id: int, item_type: "flight"|"hotel")`**: Attempts to reserve an item. Validates constraints and price freshness.
- **`Finalize()`**: Ends the episode. Success requires at least 1 flight and 1 hotel.

### Observations
The `TravelObservation` provides:
- **`itinerary`**: Current list of successful bookings.
- **`available_options`**: List of strings containing IDs, names, and prices.
- **`balance`**: Remaining funds (updates in real-time).
- **`current_goal`**: The target destination, budget, and constraints.
- **`error_log`**: Detailed feedback (e.g., "Constraint Violation", "Price expired").

---

## 📈 Baseline Performance

Performance has been significantly improved by implementing the **Hybrid State-Machine Agent** and switching to **Qwen2.5-72B-Instruct** via the Hugging Face router.

| Level | Success Rate | Final Score | Model |
| :--- | :--- | :--- | :--- |
| **1: Happy Path** | 100% | 0.950 | Qwen2.5-72B-Instruct |
| **2: Adversarial** | Adaptable | 0.070* | Qwen2.5-72B-Instruct |
| **3: Chaos** | 100% | 0.950 | Qwen2.5-72B-Instruct |

*\*Level 2 score reflects strict adherence to adversarial constraints where item availability was limited.*

---

## ⚡ Quick Start (Python)



### Minimal Code Example
```python
from travel_pro.server.env import TravelEnv
from travel_pro.models import TravelAction

env = TravelEnv()
obs = env.reset(level=1)
print(f"Goal: {obs.current_goal.destination}")

# Perform a search
action = TravelAction(type="search", query="Paris")
obs = env.step(action)
print(obs.available_options)
```

---

## 🏗️ Setup & Installation

### Local Development (via `uv`)
1. **Sync dependencies**:
   ```bash
   uv sync
   ```
2. **Run Inference Baseline**:
   ```bash
   # Set your credentials
   export HF_TOKEN="your_token_here"
   export LEVEL=ALL
   
   python3 inference.py
   ```
   *The script will output standardized logs in `[START]`, `[STEP]`, and `[END]` formats.*

### Docker Deployment
The environment is containerized for deployment on **Hugging Face Spaces**.
1. **Build the image**:
   ```bash
   docker build -t travel_pro .
   ```
2. **Run the server**:
   ```bash
   docker run -p 8000:8000 travel_pro
   ```

## 📂 Project Structure
```text
travel_pro/
├── Dockerfile             # Container definition (Production)
├── openenv.yaml           # OpenEnv manifest
├── models.py              # Pydantic V2 models
├── inference.py           # Evaluator & LLM Harness
├── database.py            # SQLAlchemy models & DB Init
└── server/
    ├── app.py             # FastAPI entry point
    └── env.py             # Core Environment Logic (Search/Book logic)
```
