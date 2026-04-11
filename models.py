from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal

class UserGoal(BaseModel):
    """Pydantic V2 model for user goals in the Travel Pro environment."""
    destination: str
    budget: float
    min_hotel_rating: float = 0.0
    is_direct_flight_required: bool = False
    max_steps: int = 10

from openenv.core.env_server.types import Action, Observation, State

class TravelObservation(Observation):
    """Pydantic V2 model for observations in the Travel Pro environment."""
    itinerary: List[str] = Field(default_factory=list)
    available_options: List[str] = Field(default_factory=list)
    balance: float
    current_goal: Optional[UserGoal] = None
    error_log: List[str] = Field(default_factory=list)
    done: bool = False

class Search(BaseModel):
    type: Literal["search"] = "search"
    query: str

class Book(BaseModel):
    type: Literal["book"] = "book"
    item_id: int
    item_type: Literal["flight", "hotel"]

class Finalize(BaseModel):
    type: Literal["finalize"] = "finalize"

from openenv.core.env_server.types import State

class TravelState(State):
    """Extended state for travel_pro environment metrics."""
    balance: float = 0.0
    error_log: List[str] = []
    itinerary_length: int = 0
    done: bool = False

class TravelAction(Action):
    """Flattened Travel Action compatible with Web UI and Inference."""
    type: Literal["search", "book", "finalize"]
    query: Optional[str] = None
    item_id: Optional[int] = None
    item_type: Optional[Literal["flight", "hotel"]] = None
