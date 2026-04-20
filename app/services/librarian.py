"""Stage 2: Librarian

Hybrid search against PostgreSQL + pgvector:
  1. Metadata filter (cuisine, max_price) as SQL WHERE clause
  2. Vector ANN search on semantic_query embedding
  3. Return top-K results
"""
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

from app.config import get_settings
from app.schema import AgentState

_engine: AsyncEngine | None = None
_embedder: SentenceTransformer | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False)
    return _engine


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


async def search(state: AgentState) -> AgentState:
    settings = get_settings()
    embedder = _get_embedder()
    engine = _get_engine()

    query_text = state.semantic_query or state.user_query
    embedding = embedder.encode(query_text).tolist()

    # Build metadata filter clauses
    filters: list[str] = []
    params: dict = {"embedding": str(embedding), "top_k": settings.top_k_results}

    if state.search_predicate:
        if state.search_predicate.cuisine:
            filters.append("cuisine ILIKE :cuisine")
            params["cuisine"] = f"%{state.search_predicate.cuisine}%"
        if state.search_predicate.max_price is not None:
            filters.append("price_range <= :max_price")
            params["max_price"] = state.search_predicate.max_price
        if state.search_predicate.min_tier is not None:
            filters.append("tier >= :min_tier")
            params["min_tier"] = state.search_predicate.min_tier

    # Always filter out tier 1 (poor) restaurants unless caller explicitly requests them
    if not any("min_tier" in f for f in filters):
        filters.append("tier >= 2")

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    sql = text(f"""
        SELECT id, name, cuisine, price_range, tier, description,
               embedding <=> CAST(:embedding AS vector) AS distance
        FROM restaurants
        {where_clause}
        ORDER BY distance
        LIMIT :top_k
    """)

    async with engine.connect() as conn:
        rows = await conn.execute(sql, params)
        state.retrieved_results = [
            {
                "id": row.id,
                "name": row.name,
                "cuisine": row.cuisine,
                "price_range": row.price_range,
                "tier": row.tier,
                "description": row.description,
                "distance": row.distance,
            }
            for row in rows
        ]

    return state
