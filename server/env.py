import random
from uuid import uuid4
from typing import List, Optional, Dict, Any, Tuple
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from travel_pro.models import UserGoal, TravelObservation, TravelAction, Search, Book, Finalize, TravelState
from travel_pro.database import SessionLocal, init_db, bulk_insert_data, Flight, Hotel
from travel_pro.scenarios import ScenarioManager

class TravelEnv(Environment):
    """
    Travel Pro Environment implementing goal-oriented travel booking.
    """
    
    def __init__(self):
        self._state = TravelState(episode_id=str(uuid4()), step_count=0)
        self.current_goal: Optional[UserGoal] = None
        self.itinerary: List[str] = []
        self.balance: float = 0.0
        self.error_log: List[str] = []
        self.price_volatility: bool = False
        self.seen_prices: Dict[int, float] = {}  # flight_id -> price at last search
        self.last_search_flights: List[Flight] = []
        self.last_search_hotels: List[Hotel] = []
        self.done: bool = False

    def _get_city_code(self, query: str) -> str:
        """Maps full city names to database codes."""
        city_map = {
            "paris": "PAR", "london": "LON", "tokyo": "TKY", "sydney": "SYD",
            "new york": "NYC", "los angeles": "LAX", "chicago": "CHI", "san francisco": "SFO"
        }
        query_l = query.lower()
        for name, code in city_map.items():
            if name in query_l or code.lower() in query_l:
                return code
        return query.upper()[:3] # Fallback

    def reset(self, level: int = 1) -> TravelObservation:
        """
        Resets the environment to a specific entropy level.
        """
        # 1. Get Scenario and Goal
        self.current_goal, env_config = ScenarioManager.get_scenario(level)
        
        # 2. Rebuild Database
        init_db()
        self._populate_db()
        
        # 3. Level 2 Special Logic: Hotel Strike
        if level == 2:
            self._apply_hotel_strike(self.current_goal.destination)
            
        # 4. Level 3 Special Logic: Price Volatility Flag
        self.price_volatility = (level == 3)
        
        self._state = TravelState(
            episode_id=str(uuid4()), 
            step_count=0,
            balance=self.balance,
            error_log=self.error_log,
            itinerary_length=0,
            done=False
        )
        self.itinerary = []
        self.balance = self.current_goal.budget
        self.error_log = ["Environment reset successful."]
        self.seen_prices = {}
        self.last_search_flights = []
        self.last_search_hotels = []
        self.done = False
        
        return self._get_obs()

    def step(self, action: TravelAction) -> TravelObservation:
        """
        Executes a step in the environment.
        Returns: observation (containing reward and done status)
        """
        assert self.current_goal is not None, "Environment must be reset before step."
        self._state.step_count += 1
        reward = -0.05  # Efficiency penalty per step
        
        # Level 3: Dynamic Price Updates
        if self.price_volatility:
            self._update_prices()
            
        step_reward = 0.0
        
        if action.type == "search":
            step_reward = self._handle_search(action)
        elif action.type == "book":
            step_reward = self._handle_book(action)
        elif action.type == "finalize":
            step_reward, self.done = self._handle_finalize()
        
        reward += step_reward
        
        # Check step limit
        if self._state.step_count >= self.current_goal.max_steps:
            if not self.done:
                reward -= 1.0  # Penalty for not finishing in time
                self.error_log.append("Deadline reached. Mission failed.")
            self.done = True
            
        info = {
            "step_count": self._state.step_count,
            "itinerary_length": len(self.itinerary)
        }
        self._state.done = self.done
        
        obs = self._get_obs()
        obs.reward = reward
        obs.done = self.done
        return obs

    def _get_obs(self) -> TravelObservation:
        """Constructs the current observation based on last search or default."""
        db = SessionLocal()
        
        # Use last search results if available, else show defaults
        flights = self.last_search_flights if self.last_search_flights else db.query(Flight).limit(5).all()
        hotels = self.last_search_hotels if self.last_search_hotels else db.query(Hotel).limit(5).all()
        
        # Ensure we have fresh instances if we just reset (optional step for stability)
        if self.last_search_flights:
            flights = db.query(Flight).filter(Flight.id.in_([f.id for f in self.last_search_flights])).all()
        if self.last_search_hotels:
            hotels = db.query(Hotel).filter(Hotel.id.in_([h.id for h in self.last_search_hotels])).all()
            
        options = [f"Flight {f.id}: {f.origin}->{f.destination} ${f.price:.2f}" for f in flights]
        options += [f"Hotel {h.id}: {h.name} in {h.city} ${h.price_per_night:.2f} Star: {h.rating}" for h in hotels]
        
        db.close()
        
        return TravelObservation(
            itinerary=self.itinerary,
            available_options=options,
            balance=self.balance,
            current_goal=self.current_goal,
            error_log=self.error_log,
            done=self.done
        )

    def _populate_db(self):
        """Initial data population using bulk_insert_data."""
        flights = []
        for i in range(20):
            flights.append({
                "origin": random.choice(["NYC", "LAX", "CHI", "SFO"]),
                "destination": random.choice(["PAR", "LON", "TKY", "SYD"]),
                "price": random.uniform(300, 1500),
                "seats_available": random.randint(1, 10),
                "is_direct": random.choice([True, False])
            })
        
        hotels = []
        cities = ["PAR", "LON", "TKY", "SYD", "NYC", "LAX", "CHI", "SFO"]
        for i in range(20):
            hotels.append({
                "city": random.choice(cities),
                "name": f"Hotel {random.randint(100, 999)}",
                "price_per_night": random.uniform(100, 600),
                "rating": random.uniform(1.0, 5.0)
            })
        
        bulk_insert_data(flights, hotels)

    def _apply_hotel_strike(self, city: str):
        """Simulates a strike by deleting 50% of hotels in the destination city."""
        db = SessionLocal()
        hotels = db.query(Hotel).filter(Hotel.city == city).all()
        if hotels:
            to_delete = hotels[:len(hotels)//2]
            for h in to_delete:
                db.delete(h)
            db.commit()
            self.error_log.append(f"HOTEL_STRIKE detected in {city}. Availability reduced.")
        db.close()

    def _update_prices(self):
        """Updates prices dynamically for Level 3 Chaos."""
        db = SessionLocal()
        flights = db.query(Flight).all()
        for f in flights:
            f.price *= (1 + random.uniform(0.01, 0.05))
        db.commit()
        db.close()

    def _handle_search(self, action: TravelAction) -> float:
        """Handles searching with SQL filtering and tracks prices for expiry checks."""
        db = SessionLocal()
        city_code = self._get_city_code(action.query)
        
        # Filter DB based on city code
        self.last_search_flights = db.query(Flight).filter(Flight.destination == city_code).limit(5).all()
        self.last_search_hotels = db.query(Hotel).filter(Hotel.city == city_code).limit(5).all()
        
        # Track prices for Chaos level
        all_flights = db.query(Flight).all()
        for f in all_flights:
            self.seen_prices[f.id] = f.price
        db.close()
        
        self.error_log.append(f"Search for '{action.query}' (Code: {city_code}) found {len(self.last_search_flights)} flights and {len(self.last_search_hotels)} hotels.")
        return 0.0

    def _handle_book(self, action: TravelAction) -> float:
        """Handles booking with price expiry and constraint validation."""
        db = SessionLocal()
        reward = 0.0
        
        if action.item_type == "flight":
            item = db.query(Flight).filter(Flight.id == action.item_id).first()
            if not item:
                self.error_log.append(f"Flight {action.item_id} not found.")
                return -0.1
            
            # Price Expiry Check (Chaos) - Added 10% buffer to allow for step-by-step increases
            if self.price_volatility and action.item_id in self.seen_prices:
                max_allowed_price = self.seen_prices[action.item_id] * 1.10
                if item.price > max_allowed_price:
                    self.error_log.append(f"Price expired. Current: ${item.price:.2f} > Max: ${max_allowed_price:.2f}. Re-search required.")
                    db.close()
                    return -0.1

            # Constraint Validation (Direct Flight)
            if self.current_goal.is_direct_flight_required and not item.is_direct:
                reward -= 1.0
                self.error_log.append("Constraint Violation: Booked indirect flight.")

            if self.balance >= item.price and item.seats_available > 0:
                item.seats_available -= 1
                self.balance -= item.price
                self.itinerary.append(f"Flight to {item.destination}")
                db.commit()
            else:
                self.error_log.append("Booking failed: Insufficient funds or seats.")
                reward -= 0.1

        elif action.item_type == "hotel":
            item = db.query(Hotel).filter(Hotel.id == action.item_id).first()
            if not item:
                self.error_log.append(f"Hotel {action.item_id} not found.")
                return -0.1
            
            # Constraint Validation (Rating)
            if item.rating < self.current_goal.min_hotel_rating:
                reward -= 1.0
                self.error_log.append(f"Constraint Violation: Hotel rating {item.rating} < {self.current_goal.min_hotel_rating}")

            if self.balance >= item.price_per_night:
                self.balance -= item.price_per_night
                self.itinerary.append(f"Hotel in {item.city}")
                db.commit()
            else:
                self.error_log.append("Booking failed: Insufficient funds.")
                reward -= 0.1
                
        db.close()
        return reward

    def _handle_finalize(self) -> Tuple[float, bool]:
        """Finalizes the trip and awards completion reward."""
        if len(self.itinerary) >= 2:  # Assume at least a flight and a hotel for success
            self.error_log.append("Trip finalized. Goal achieved!")
            return 1.0, True
        else:
            self.error_log.append("Trip finalized prematurely. Goal failed.")
            return -0.5, True

    @property
    def state(self) -> State:
        return self._state
