from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List


class SearchPredicate(BaseModel):
    cuisine: Optional[str] = None
    max_price: Optional[float] = None
    min_tier: Optional[int] = None   # 1-4; Librarian filters out restaurants below this


class DispatcherOutput(BaseModel):
    """Structured output from the Dispatcher (Stage 1).
    Instructor forces the LLM to return exactly this schema.
    """
    persona: Literal["foodie", "normie", "neutral"] = "neutral"
    attack: bool = False
    search_predicate: Optional[SearchPredicate] = None
    semantic_query: Optional[str] = None


class AgentState(BaseModel):
    """Single source of truth that flows through the pipeline."""

    # --- Input ---
    user_query: str

    # --- Stage 1: Dispatcher output ---
    persona: Literal["foodie", "normie", "neutral"] = "neutral"
    attack: bool = False
    search_predicate: Optional[SearchPredicate] = None
    semantic_query: Optional[str] = None

    # --- Stage 2: Librarian output ---
    retrieved_results: List[Dict] = Field(default_factory=list)

    # --- Stage 3: Concierge output ---
    suggestion: str = ""
    elaboration: str = ""


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    suggestion: str
    elaboration: str
    persona: str
    attack: bool
