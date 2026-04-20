# Restaurant Concierge Agent (RCA)

A 3-stage LLM inference pipeline that takes a natural-language restaurant request and returns a persona-adaptive recommendation, backed by a vector database of restaurant records.

---

## Architecture

```
User Query
    │
    ▼
[Stage 1: Dispatcher]  ──── Fine-tuned Llama 3.2 1B (QLoRA)
  Extracts: persona, attack flag, search_predicate, semantic_query
    │
    ├─ attack=True ──► Safe refusal response (pipeline short-circuits)
    │
    ▼
[Stage 2: Librarian]  ──── PostgreSQL + pgvector
  Hybrid search: metadata filter (cuisine, price, tier) + ANN vector search
    │
    ▼
[Stage 3: Concierge]  ──── Llama 3.2 3B
  Generates persona-adaptive narrative (foodie / normie / neutral)
  Factors in quality tier: calls out award-winners, warns on poor options
```

**Persona logic:** The Dispatcher classifies users as `foodie` (culinary terminology), `normie` (casual), or `neutral`. The Concierge mirrors this — foodies get sensory/technique language, normies get plain practical info.

**Attack detection:** Prompt injection attempts are classified by the Dispatcher (`attack=true`) and short-circuit the pipeline, returning a safe canned response without touching the database or Concierge.

**Quality tiers:** Each restaurant has a `tier` (1–4). The Dispatcher infers a `min_tier` filter from language cues. The Concierge surfaces tier 4 accolades and warns honestly on tier 1/2.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Package manager | uv |
| API framework | FastAPI + asyncio |
| Schema / validation | Pydantic v2 + Instructor |
| Inference server | vLLM (OpenAI-compatible, 4-bit AWQ) |
| Fine-tuning | Unsloth + QLoRA |
| Vector DB | PostgreSQL + pgvector |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Stage 1 model | Llama 3.2 1B Instruct (fine-tuned) |
| Stage 3 model | Llama 3.2 3B Instruct |
| Container runtime | Podman (k8s YAML pod) |

---

## Project Structure

```
rca/
  app/
    schema.py               # AgentState, DispatcherOutput, SearchPredicate
    config.py               # Settings (pydantic-settings), inference params
    main.py                 # FastAPI app — /health, /query
    services/
      dispatcher.py         # Stage 1: vLLM + Instructor structured extraction
      librarian.py          # Stage 2: pgvector hybrid search
      concierge.py          # Stage 3: persona-adaptive narrative generation
  data/
    synthetic/
      train.jsonl           # 1000 training samples (800 normal + 200 attack)
      eval.jsonl            # 100 held-out evaluation samples
    restaurant_data/
      synthetic_restaurants.json   # 200 restaurants with tier ratings
  scripts/
    generate_dataset_bedrock.py   # Generate training data via AWS Bedrock
    backfill_tiers.py             # Assign quality tiers + rewrite descriptions
    ingest.py                     # Embed + load restaurants into pgvector
    ingest_bedrock.py             # Generate synthetic restaurant data via Bedrock
    finetune.py                   # QLoRA fine-tune Llama 3.2 1B via Unsloth
    train_dispatcher.py           # Generate data → rsync → finetune → evaluate Dispatcher
    train_concierge.py            # Generate data → rsync → finetune → evaluate Concierge
    evaluate.py                   # Field-level F1 evaluation of Dispatcher
    check_deps.sh                 # Pre-flight checks on GPU machine
    deploy.sh                     # Build + launch Podman pod
    db_init.sql                   # PostgreSQL schema
  deploy/
    pod.yaml                      # Podman k8s-style pod definition
    pod.env                       # Pod config variables (paths, ports)
  tests/
    test_pipeline.py              # Integration tests
  Dockerfile                      # API container image
  pyproject.toml                  # uv-managed dependencies
  .env.example                    # Environment variable template
```

---

## Quickstart (GPU Machine)

### Prerequisites

```bash
bash scripts/check_deps.sh   # verify podman, nvidia-ctk, CUDA, ports
```

### 1. Configure environment

```bash
cp .env.example .env
# Fill in: HUGGING_FACE_HUB_TOKEN, ANTHROPIC_API_KEY
```

### 2. Deploy the pod

```bash
bash scripts/deploy.sh
```

This builds the API image, resolves paths in `deploy/pod.yaml`, and launches the pod with vLLM + PostgreSQL + FastAPI.

### 3. Initialize the database

```bash
# Wait ~30s for postgres to be ready, then:
podman exec rca-pod-postgres psql -U rca -d rca -f /scripts/db_init.sql
uv run scripts/ingest.py --source data/restaurant_data/synthetic_restaurants.json
```

### 4. Generate training data (Mac — uses AWS Bedrock)

```bash
uv run scripts/generate_dataset_bedrock.py       # Dispatcher: 1000 train + 100 eval
uv run scripts/generate_concierge_dataset.py     # Concierge: 500 train + 60 eval
```

### 5. Train the models (run directly on GPU machine)

```bash
uv run scripts/train_dispatcher.py    # finetune Llama 3.2 1B → evaluate
uv run scripts/train_concierge.py     # finetune Llama 3.2 3B → evaluate
```

### 5. Test the API

```bash
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I want a proper omakase, money is no object"}' | jq .

curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "cheap tacos near me, nothing fancy"}' | jq .
```

---

## Inference Parameters

Documented for rubric compliance.

| Parameter | Dispatcher | Concierge |
|---|---|---|
| Model | Llama 3.2 1B (fine-tuned) | Llama 3.2 3B Instruct |
| Quantization | 4-bit AWQ via vLLM | 4-bit AWQ via vLLM |
| Temperature | 0.1 (deterministic extraction) | 0.7 (creative narrative) |
| Top-p | — | 0.9 |
| Max tokens | 256 | 512 |
| Sampling mode | Greedy (low temp) | Nucleus sampling |

---

## Training Data

| File | Samples | Description |
|---|---|---|
| `data/synthetic/train.jsonl` | 1000 | 800 normal queries + 200 attack samples |
| `data/synthetic/eval.jsonl` | 100 | Held-out normal queries (no attacks) |

Each sample: `{"input": "<user query>", "output": {"persona": "...", "attack": bool, "search_predicate": {...}, "semantic_query": "..."}}`

Attack samples use deterministic surface mutations (case changes, wrapper prefixes, unicode substitutions) to ensure variety without triggering model safety refusals during generation.

---

## Evaluation

```bash
# Run on GPU machine after fine-tuning and vLLM serving the fine-tuned model:
uv run scripts/evaluate.py --model-name dispatcher-llama-1b
```

**Metrics:**
- `persona_accuracy` — exact match (3-class)
- `attack_accuracy` — exact match (binary)
- `cuisine_precision` — exact match on non-null cuisine labels
- `price_mae` — mean absolute error on non-null max_price predictions

Results written to `data/synthetic/eval_results_dispatcher-llama-1b.json`.

---

## Restaurant Data

200 synthetic restaurants across 12 cuisines, generated via Claude (Bedrock).

| Tier | Count | Description |
|---|---|---|
| 4 | 30 | Award-winning, Michelin/James Beard caliber |
| 3 | 110 | Solid neighborhood favorite |
| 2 | 40 | Mixed — one strength, one notable flaw |
| 1 | 20 | Poor — overpriced, declining, or disappointing |

---

## API Reference

### `POST /query`

```json
// Request
{"query": "I want cheap Italian food under $20"}

// Response
{
  "suggestion": "...",
  "elaboration": "...",
  "persona": "normie",
  "attack": false
}
```

### `GET /health`

```json
{"status": "ok"}
```

Interactive docs available at `http://localhost:8080/docs`.
