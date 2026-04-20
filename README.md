# Restaurant Concierge Agent (RCA)

A 3-stage LLM inference pipeline that takes a natural-language restaurant request and returns a persona-adaptive recommendation, backed by a vector database of restaurant records.

---

## Architecture

```
User Query
    │
    ▼
[Stage 1: Dispatcher]  ──── Fine-tuned Llama 3.2 1B (QLoRA) + Structured Outputs
  Extracts: persona, attack flag, search_predicate, semantic_query
    │
    ├─ attack=True ──► Safe refusal response (pipeline short-circuits)
    │
    ▼
[Stage 2: Librarian]  ──── PostgreSQL + pgvector
  Hybrid search: metadata filter (cuisine, price, tier) + ANN vector search
    │
    ▼
[Stage 3: Concierge]  ──── Fine-tuned Llama 3.2 3B (QLoRA)
  Generates persona-adaptive narrative (foodie / normie / neutral)
  Grounded strictly in retrieved candidates — no hallucination
```

**Persona logic:** The Dispatcher classifies users as `foodie` (culinary terminology), `normie` (casual), or `neutral`. The Concierge mirrors this — foodies get sensory/technique language, normies get plain practical info.

**Attack detection:** Prompt injection attempts are classified by the Dispatcher (`attack=true`) and short-circuit the pipeline, returning a safe canned response without touching the database or Concierge. Constrained decoding (vLLM `structured_outputs`) ensures 100% attack recall.

**Quality tiers:** Each restaurant has a `tier` (1–4). The Dispatcher infers a `min_tier` filter from language cues. The Concierge surfaces tier 4 accolades and warns on tier 2 trade-offs.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Package manager | uv |
| API framework | FastAPI + asyncio |
| Schema / validation | Pydantic v2 |
| Inference server | vLLM (OpenAI-compatible, 4-bit bitsandbytes) |
| Structured outputs | vLLM `structured_outputs` (Outlines constrained decoding) |
| Fine-tuning | Unsloth + QLoRA |
| Vector DB | PostgreSQL + pgvector |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Stage 1 model | Llama 3.2 1B Instruct (fine-tuned) |
| Stage 3 model | Llama 3.2 3B Instruct (fine-tuned) |
| UI | Streamlit (conversational chat interface) |
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
      dispatcher.py         # Stage 1: vLLM + structured_outputs constrained decoding
      librarian.py          # Stage 2: pgvector hybrid search
      concierge.py          # Stage 3: persona-adaptive narrative generation
  ui/
    app.py                  # Streamlit chat UI
  data/
    synthetic/
      train.jsonl           # ~990 Dispatcher training samples
      eval.jsonl            # ~47 held-out evaluation samples
      concierge_train.jsonl # 552 pipeline-driven Concierge training samples
      concierge_eval.jsonl  # 25 held-out Concierge eval samples
    restaurant_data/
      synthetic_restaurants.json   # 200 restaurants with tier ratings
  scripts/
    generate_dataset_bedrock.py        # Generate Dispatcher training data via AWS Bedrock
    generate_dispatcher_dataset_v2.py  # Improved Dispatcher dataset (subtle attacks)
    generate_concierge_dataset_mac.py  # Pipeline-driven Concierge dataset (Mac + Bedrock)
    backfill_tiers.py                  # Assign quality tiers + rewrite descriptions
    ingest.py                          # Embed + load restaurants into pgvector
    finetune.py                        # QLoRA fine-tune Dispatcher (Llama 3.2 1B)
    finetune_concierge.py              # QLoRA fine-tune Concierge (Llama 3.2 3B)
    patch_tokenizers.py                # Fix Unsloth tokenizer for vLLM compatibility
    train_dispatcher.py                # Finetune + evaluate Dispatcher (runs on GPU)
    train_concierge.py                 # Finetune + evaluate Concierge (runs on GPU)
    evaluate.py                        # Field-level F1 evaluation — Dispatcher
    evaluate_concierge.py              # Generative metrics evaluation — Concierge
    evaluate_pipeline.py               # End-to-end pipeline evaluation (any endpoint)
    check_deps.sh                      # Pre-flight checks on GPU machine
    setup_gpu_machine.sh               # One-time GPU machine setup
    deploy_native_vllm.sh              # Deploy vLLM natively + pod (current)
    deploy.sh                          # Full pod deploy (requires Podman 5.x + CDI)
    db_init.sql                        # PostgreSQL schema
  deploy/
    pod.yaml                # Podman pod — postgres + api + ui (current)
    pod_full.yaml           # Future: full pod with vLLM containers (Podman 5.x)
    pod.env                 # Pod config variables (paths, ports)
    cloudrun-api.yaml       # GCP Cloud Run — API service
    cloudrun-ui.yaml        # GCP Cloud Run — UI service
    gcp-setup.sh            # One-time GCP infrastructure setup
    cloudsql-init.sql       # Cloud SQL schema (pgvector notes)
  .github/
    workflows/
      deploy-gcp.yml        # GitHub Actions CI/CD → Artifact Registry → Cloud Run
  docs/
    architecture.md         # System architecture and data flow
    models.md               # Model selection, fine-tuning, evaluation
    inference.md            # Sampling parameters, quantization, vLLM config
    deployment.md           # Local and GCP deployment guides
    api.md                  # API reference with curl examples
    evaluation_report.md    # Pipeline evaluation results across configurations
  tests/                    # 164 unit tests (pytest)
  Dockerfile                # API container image
  Dockerfile.ui             # Streamlit UI container image
  Dockerfile.vllm           # vLLM container image (for full pod / GCP)
  pyproject.toml            # uv-managed dependencies
  .env.example              # Environment variable template
```

---

## Quickstart (GPU Machine)

### Prerequisites

```bash
# One-time machine setup (installs nvidia-ctk, configures CDI)
bash scripts/setup_gpu_machine.sh

# Verify everything is ready
bash scripts/check_deps.sh
```

### 1. Configure environment

```bash
cp .env.example .env
# Fill in: HUGGING_FACE_HUB_TOKEN
```

### 2. Deploy the stack

```bash
# Starts vLLM natively (ports 8000 + 8001) then launches pod (postgres + api + ui)
bash scripts/deploy_native_vllm.sh
```

The script: stops existing services → checks dependencies → rotates logs → builds images → starts vLLM (waits for health) → launches pod.

### 3. Initialize the database

```bash
# Wait ~30s for postgres to be ready, then:
podman exec rca-pod-postgres psql -U rca -d rca -f /scripts/db_init.sql
uv run scripts/ingest.py --source data/restaurant_data/synthetic_restaurants.json
```

### 4. Test the API

```bash
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I want a proper omakase, money is no object"}' | jq .

curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "cheap tacos, nothing fancy"}' | jq .
```

### 5. Open the UI

Navigate to `http://localhost:8501` for the Streamlit chat interface.

---

## Training Workflow

Training data is generated on the Mac (uses AWS Bedrock), then synced to the GPU machine for fine-tuning.

```bash
# On Mac — generate training data
uv run scripts/generate_dataset_bedrock.py          # Dispatcher: ~990 train + ~47 eval
uv run scripts/generate_concierge_dataset_mac.py    # Concierge: 552 train + 25 eval

# Sync to GPU machine
bash sync.sh --push

# On GPU machine — fine-tune (vLLM must be stopped first to free VRAM)
uv run scripts/train_dispatcher.py   # QLoRA finetune Llama 3.2 1B → evaluate
uv run scripts/train_concierge.py    # QLoRA finetune Llama 3.2 3B → evaluate

# Patch tokenizer after fine-tuning (Unsloth tokenizer compatibility fix)
uv run scripts/patch_tokenizers.py
```

---

## Inference Parameters

| Parameter | Dispatcher | Concierge |
|---|---|---|
| Model | Llama 3.2 1B (fine-tuned) | Llama 3.2 3B (fine-tuned) |
| Quantization | 4-bit bitsandbytes | 4-bit bitsandbytes |
| Structured outputs | `structured_outputs: {json: schema}` | `response_format: json_object` |
| Temperature | 0.1 (deterministic extraction) | 0.7 (creative narrative) |
| Top-p | — | 0.9 |
| Max tokens | 256 | 512 |
| Sampling mode | Greedy + constrained | Nucleus sampling |
| GPU memory utilization | 0.40 | 0.50 |
| Max model length | 2048 | 2048 |

---

## Evaluation

```bash
# Individual model evaluation (on GPU machine, vLLM running)
uv run scripts/evaluate.py              # Dispatcher field-level F1
uv run scripts/evaluate_concierge.py    # Concierge generative metrics

# End-to-end pipeline evaluation (from any machine with network access)
uv run scripts/evaluate_pipeline.py --base-url http://192.168.200.100:8080

# Compare local vs GCP deployment
uv run scripts/evaluate_pipeline.py \
  --base-url http://192.168.200.100:8080 \
  --compare-url https://<cloud-run-url>
```

See `docs/evaluation_report.md` for full results across all configurations.

---

## Restaurant Data

200 synthetic restaurants across 12 cuisines, generated via Claude (Bedrock) with quality tier ratings.

| Tier | Count | Description |
|---|---|---|
| 4 | 30 | Award-winning, Michelin/James Beard caliber |
| 3 | 110 | Solid neighborhood favorite |
| 2 | 40 | Mixed — one strength, one notable flaw |
| 1 | 20 | Poor — overpriced, declining, or disappointing |

Tier 1 restaurants are filtered out by the Librarian by default (`tier >= 2`).

---

## API Reference

### `POST /query`

```json
// Request
{"query": "I want cozy Italian for date night under $60"}

// Response
{
  "suggestion": "San Gennaro delivers exactly what you want...",
  "elaboration": "This Neapolitan trattoria features...",
  "persona": "normie",
  "attack": false
}
```

### `GET /health`

```json
{"status": "ok"}
```

Interactive docs: `http://localhost:8080/docs`

---

## Documentation

| File | Contents |
|---|---|
| `docs/architecture.md` | System design, data flow, AgentState schema |
| `docs/models.md` | Model selection, fine-tuning methodology, training config, evaluation metrics |
| `docs/inference.md` | Sampling parameters, quantization, vLLM configuration |
| `docs/deployment.md` | Local and GCP deployment step-by-step |
| `docs/api.md` | Full API reference with curl examples |
| `docs/evaluation_report.md` | Pipeline evaluation results — v1, v2, v1+structured_outputs comparison |
