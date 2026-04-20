from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schema import AgentState, QueryRequest, QueryResponse
from app.services import dispatcher, librarian, concierge


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model on startup
    from app.services.librarian import _get_embedder
    _get_embedder()
    yield


app = FastAPI(
    title="Restaurant Concierge Agent",
    description="3-stage LLM inference pipeline: Dispatcher → Librarian → Concierge",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    state = AgentState(user_query=request.query)

    # Stage 1: Extract intent
    state = await dispatcher.run(state)

    # Stage 2: Retrieve candidates (skip if attack)
    if not state.attack:
        state = await librarian.search(state)

    # Stage 3: Generate narrative
    state = await concierge.synthesize(state)

    return QueryResponse(
        suggestion=state.suggestion,
        elaboration=state.elaboration,
        persona=state.persona,
        attack=state.attack,
    )
