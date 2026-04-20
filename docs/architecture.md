# System Architecture

## Overview

The Restaurant Concierge Agent (RCA) is a 3-stage LLM inference pipeline that accepts a natural-language restaurant query and returns a persona-adaptive recommendation backed by a pgvector database of 200 restaurant records.

---

## System Architecture Diagram

```
                         ┌─────────────────────────────────────────┐
                         │             Client (HTTP)                │
                         └──────────────────┬──────────────────────┘
                                            │ POST /query {"query": "..."}
                                            ▼
                         ┌─────────────────────────────────────────┐
                         │           FastAPI Application           │
                         │             app/main.py                 │
                         │       (asyncio, port 8080)              │
                         └──────────────────┬──────────────────────┘
                                            │ AgentState(user_query)
                                            ▼
                  ┌─────────────────────────────────────────────────┐
                  │              Stage 1: Dispatcher                │
                  │         app/services/dispatcher.py              │
                  │   Fine-tuned Llama 3.2 1B (QLoRA) via vLLM     │
                  │   Input:  user_query (raw string)               │
                  │   Output: persona, attack, search_predicate,    │
                  │           semantic_query                        │
                  └──────────┬──────────────────────┬──────────────┘
                             │                      │
                    attack=False               attack=True
                             │                      │
                             ▼                      ▼
          ┌──────────────────────────┐   ┌──────────────────────────┐
          │    Stage 2: Librarian    │   │   Short-Circuit Response  │
          │  app/services/librarian  │   │  "I'm here to help with  │
          │  PostgreSQL + pgvector   │   │  restaurant recomm..."    │
          │  Hybrid search:          │   │  (no DB or model call)   │
          │  - metadata WHERE clause │   └──────────────────────────┘
          │  - ANN cosine similarity │
          │  Output: retrieved_results│
          └──────────┬───────────────┘
                     │ top-K restaurant records
                     ▼
          ┌──────────────────────────────────────────────┐
          │             Stage 3: Concierge               │
          │         app/services/concierge.py            │
          │         Llama 3.2 3B Instruct via vLLM       │
          │   Input:  retrieved_results + persona +      │
          │           user_query                         │
          │   Output: suggestion + elaboration           │
          └──────────┬───────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────────────────────────────┐
          │         QueryResponse (JSON)                 │
          │  suggestion, elaboration, persona, attack    │
          └──────────────────────────────────────────────┘
```

### Infrastructure Layout

```
GPU Machine (192.168.200.100 — Linux, RTX 3060)
├── vLLM Dispatcher     — native process, port 8000
├── vLLM Concierge      — native process, port 8001
└── rca-pod (Podman)
    ├── rca-pod-postgres — pgvector DB, port 5432
    └── rca-pod-api      — FastAPI app, port 8080
```

---

## 3-Stage Pipeline Explanation

### Stage 1: Dispatcher (`app/services/dispatcher.py`)

The Dispatcher is the entry point for every request. It runs a fine-tuned Llama 3.2 1B model to transform raw user text into a structured `DispatcherOutput` object.

**Responsibilities:**
- Classify user persona (`foodie` / `normie` / `neutral`)
- Detect prompt injection attacks (`attack: bool`)
- Extract a structured metadata filter (`search_predicate`) with optional `cuisine`, `max_price`, and `min_tier` fields
- Generate a `semantic_query` string for vector similarity search

**Implementation details:**
- Calls vLLM (port 8000) via the OpenAI-compatible chat completions API
- Uses `json.JSONDecoder().raw_decode()` to extract the first valid JSON object from model output, handling cases where the model emits multiple objects
- Falls back to a safe default `DispatcherOutput()` on any parse failure — the pipeline never crashes due to malformed model output
- Temperature is set to 0.1 for near-deterministic extraction (see `app/config.py`)

### Stage 2: Librarian (`app/services/librarian.py`)

The Librarian performs hybrid search against a PostgreSQL + pgvector database to retrieve relevant restaurant candidates.

**Responsibilities:**
- Embed the `semantic_query` (or `user_query` as fallback) using `sentence-transformers/all-MiniLM-L6-v2`
- Apply metadata filters as SQL WHERE clauses (cuisine ILIKE, price_range <=, tier >=)
- Execute approximate nearest-neighbor (ANN) cosine similarity search via the `<=>` pgvector operator
- Return the top-K results ranked by embedding distance

**Lazy initialization:** The embedding model and database engine are initialized once on first call and reused across requests (module-level singletons).

### Stage 3: Concierge (`app/services/concierge.py`)

The Concierge synthesizes retrieved candidates into a natural-language recommendation tailored to the detected persona.

**Responsibilities:**
- Select the appropriate persona instruction (sensory/culinary vs. plain/casual vs. balanced)
- Inject the candidate list and tier legend into the system prompt
- Call Llama 3.2 3B via vLLM (port 8001) with `response_format={"type": "json_object"}` for structured output
- Parse `suggestion` and `elaboration` from the JSON response
- Enforce the no-hallucination constraint: the model is strictly instructed to only reference restaurants from the provided candidate list

---

## Attack Detection and Short-Circuit Behavior

The Dispatcher classifies any message as an attack if it contains:
- Prompt injection attempts (e.g., "Ignore previous instructions...")
- Jailbreak attempts
- Instruction overrides or role-play hijacks
- Any attempt to subvert the system's restaurant recommendation function

When `attack=True` is returned by the Dispatcher, the pipeline **short-circuits immediately**:

```
dispatcher.run(state)  →  state.attack = True
    ↓ (no Librarian call)
    ↓ (no database query)
concierge.synthesize(state)  →  returns SAFE_RESPONSE directly
    ↓
QueryResponse(suggestion=SAFE_RESPONSE, elaboration="", attack=True)
```

The database is never queried, and the Concierge model is not invoked. This prevents:
- Data exfiltration via crafted prompts
- Generation of off-topic content
- Resource consumption from adversarial inputs

The training dataset contains ~200 attack samples (20% of total) covering jailbreak patterns, case mutations, unicode substitutions, and wrapper prefixes to ensure the Dispatcher is robust to diverse attack surfaces.

---

## Persona Classification

The Dispatcher classifies users into one of three personas based on vocabulary and phrasing:

| Persona | Signal | Concierge Behavior |
|---------|--------|-------------------|
| `foodie` | Culinary terminology: umami, terroir, mise en place, maillard, tasting menu, omakase | Rich sensory language; highlights technique, chef philosophy, ingredient detail, flavor profiles |
| `normie` | Casual phrasing: "cheap", "good vibes", "nothing fancy", "near me" | Plain language; focuses on vibe, value, wait times, consensus opinion |
| `neutral` | Ambiguous or formal requests | Balanced, factual recommendation without jargon |

The persona is extracted by Stage 1 and passed directly to Stage 3. Stage 2 (Librarian) is persona-agnostic — it retrieves the same candidates regardless of persona.

---

## Quality Tier System (1–4)

Each restaurant record in the database carries a `tier` field (1–4) representing overall quality:

| Tier | Count | Meaning |
|------|-------|---------|
| 4 | 30 | Award-winning — Michelin/James Beard caliber |
| 3 | 110 | Solid neighborhood favorite |
| 2 | 40 | Mixed — one strength, one notable flaw |
| 1 | 20 | Poor — overpriced, declining, or disappointing |

**Dispatcher tier inference:** The Dispatcher extracts a `min_tier` value from language cues in the query:
- `min_tier=4` — "best in the city", "award-winning", "Michelin-level"
- `min_tier=3` — "reliable", "solid", "good" (default for most queries)
- `min_tier=2` — "doesn't have to be fancy", "just decent", "trade-offs are fine"
- `min_tier=1` — only if the user explicitly wants cheapest regardless of quality

**Librarian default filter:** If no `min_tier` is specified in the search predicate, the Librarian automatically filters `tier >= 2`, preventing recommendation of known poor-quality restaurants unless explicitly requested.

**Concierge tier awareness:** The system prompt instructs the Concierge to:
- Surface tier 4 accolades prominently
- Name the specific trade-off for any tier 2 recommendation
- Never silently recommend a tier 1 restaurant

---

## Data Flow: AgentState

`AgentState` (defined in `app/schema.py`) is the single mutable object passed between all three stages. It is a Pydantic model that accumulates results as each stage runs.

```python
class AgentState(BaseModel):
    # Input (from HTTP request)
    user_query: str

    # Populated by Stage 1 (Dispatcher)
    persona: Literal["foodie", "normie", "neutral"] = "neutral"
    attack: bool = False
    search_predicate: Optional[SearchPredicate] = None   # cuisine, max_price, min_tier
    semantic_query: Optional[str] = None

    # Populated by Stage 2 (Librarian)
    retrieved_results: List[Dict] = []   # top-K restaurant records with distance scores

    # Populated by Stage 3 (Concierge)
    suggestion: str = ""
    elaboration: str = ""
```

Each stage receives the full `AgentState`, mutates its designated fields, and returns it. This makes the pipeline trivially composable and the state at any point in execution fully inspectable.

The `QueryResponse` returned to the client is a projection of `AgentState`: `suggestion`, `elaboration`, `persona`, and `attack`.
