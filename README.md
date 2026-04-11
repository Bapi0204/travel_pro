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
1.  **Level 1 (Happy Path)**: High budget ($5,000), stable prices, and infinite availability. Testing basic tool-calling.
2.  **Level 2 (Adversarial)**: Injects **HOTEL_STRIKE** (low availability) or **FLIGHT_CRUNCH** (high prices). Strict constraints: must be 4+ stars and direct flights only.
3.  **Level 3 (Chaos Mode)**: Activates **Price Volatility**. Prices increase by 1-5% every single step. Agents must handle `Price Expired` errors by re-searching data.

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

Performance has been significantly improved by switching to **GPT-4o** and implementing **Intelligent Search Filtering**.

| Level | Success Rate | Avg Steps | Final Score | Model |
| :--- | :--- | :--- | :--- | :--- |
| **1: Happy Path** | 100% | 3.0 | 0.960 | GPT-4o |
| **2: Adversarial** | 80% | 5.2 | 0.820 | GPT-4o |
| **3: Chaos** | 60% | 8.5 | 0.650 | GPT-4o |

---

## ⚡ Quick Start (Python)

If you want to quickly test the environment logic without running a full LLM agent, use the `demo.py` script:

```bash
uv run python demo.py
```

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
   uv run python inference.py
   ```

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
