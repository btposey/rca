# Inference Pipeline

## Overview

The inference pipeline consists of three components running on a single GPU machine (RTX 3060, 12 GB VRAM):
- **Two vLLM inference servers** (Dispatcher on port 8000, Concierge on port 8001) running as native host processes
- **One pgvector search stage** (Librarian) running against a Podman-managed PostgreSQL container

All three stages execute sequentially within a single async request handler in FastAPI.

---

## Sampling Parameters

Sampling parameters are defined in `app/config.py` and documented here for rubric compliance.

### Dispatcher (Stage 1) — Deterministic Extraction

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | **0.1** | Near-deterministic; extraction tasks benefit from greedy-like behavior — the correct JSON output is unique, not creative |
| `max_tokens` | **256** | Sufficient for the JSON output schema; caps token cost for extraction |
| `top_p` | not set | Nucleus sampling not used; low temperature makes top_p redundant |
| Sampling mode | Greedy (effectively) | Low temperature collapses the distribution to near-argmax selection |

**Rationale for temperature=0.1:** The Dispatcher maps a user string to a fixed JSON schema. There is one correct answer per input. High temperature introduces noise into extraction, producing field values that drift from ground truth. Temperature 0.1 keeps the model on the high-probability path while avoiding the occasional degenerate outputs that pure greedy (temperature=0) can produce.

### Concierge (Stage 3) — Creative Narrative

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | **0.7** | Moderate creativity; produces varied, natural-sounding prose without excessive randomness |
| `top_p` | **0.9** | Nucleus sampling: considers only tokens whose cumulative probability reaches 90%, filtering the long tail of low-probability vocabulary |
| `max_tokens` | **512** | Covers a `suggestion` (1–2 sentences) + `elaboration` (2–4 sentences) with room for JSON framing |
| Sampling mode | Nucleus (top-p) | Standard setting for open-ended generation balancing quality and diversity |

**Rationale for temperature=0.7 + top_p=0.9:** The Concierge generates restaurant recommendations that need to feel natural and slightly different across requests (not robotic repetitions of the same phrasing). Temperature 0.7 with top_p=0.9 is a well-established default for instruction-following generation tasks — creative enough to vary, conservative enough to stay coherent.

### Configuration Source

```python
# app/config.py
class Settings(BaseSettings):
    dispatcher_temperature: float = 0.1   # low — deterministic extraction
    dispatcher_max_tokens: int = 256
    concierge_temperature: float = 0.7    # higher — creative narrative
    concierge_top_p: float = 0.9
    concierge_max_tokens: int = 512
```

These values are overridable via environment variables (e.g., `CONCIERGE_TEMPERATURE=0.5`).

---

## Quantization: 4-bit via bitsandbytes

Both models are served in **4-bit quantization** using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) via vLLM's `--quantization bitsandbytes` flag.

**What this does:**
- Represents model weights in NF4 (4-bit NormalFloat) format instead of fp16/bf16
- Reduces model memory footprint by ~4x (e.g., Llama 3.2 3B: ~6 GB fp16 → ~1.5 GB 4-bit)
- Enables two separate vLLM instances (Dispatcher + Concierge) to coexist on a single 12 GB GPU

**Tradeoffs:**
- Minor quality degradation from quantization noise (~1–3% on benchmarks for 1B/3B models)
- Inference throughput lower than fp16 (dequantization overhead), but acceptable for low-concurrency local deployment
- 4-bit bitsandbytes is compatible with RTX 3060 (Ampere architecture, CUDA 11.x+)

**Why not AWQ/GPTQ?**
AWQ and GPTQ require pre-quantized model checkpoints or a separate calibration step. The `--quantization bitsandbytes` flag quantizes on the fly during model load, which is simpler for locally trained models that do not have pre-computed quantization scales.

---

## vLLM Configuration

Each vLLM instance is launched with the following flags (from `scripts/deploy_native_vllm.sh`):

### Dispatcher Instance (port 8000)

```bash
vllm serve ${MODEL_HOST_PATH}/dispatcher-llama-1b \
    --served-model-name dispatcher-llama-1b \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization bitsandbytes \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.40 \
    --max-num-seqs 8
```

### Concierge Instance (port 8001)

```bash
vllm serve ${MODEL_HOST_PATH}/concierge-llama-3b \
    --served-model-name concierge-llama-3b \
    --host 0.0.0.0 \
    --port 8001 \
    --quantization bitsandbytes \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.50
```

### Parameter Rationale

| Parameter | Dispatcher | Concierge | Rationale |
|-----------|-----------|-----------|-----------|
| `--max-model-len` | 2048 | 2048 | Maximum sequence length. Limits KV cache allocation. 2048 covers all realistic inputs; longer contexts would require more VRAM. |
| `--gpu-memory-utilization` | 0.40 | 0.50 | Fraction of GPU VRAM allocated to this instance. Split is 40/50 to leave ~10% headroom. Dispatcher gets less because its inputs/outputs are shorter. |
| `--max-num-seqs` | 8 | (default) | Maximum concurrent sequences in the Dispatcher's continuous batching scheduler. Capped at 8 because the Dispatcher handles very short sequences and high parallelism would spike VRAM. |
| `--quantization bitsandbytes` | both | both | 4-bit on-the-fly quantization (see above). |

**GPU memory split:** 0.40 + 0.50 = 0.90 of 12 GB VRAM = ~10.8 GB allocated across both models, leaving ~1.2 GB for OS and CUDA overhead. This split is tuned for RTX 3060; adjust `gpu_memory_utilization` if deploying on different hardware.

---

## Structured Output Approach

### Dispatcher: JSON Extraction with Fallback

The Dispatcher uses a two-layer approach to guarantee valid structured output:

**Layer 1 — JSON extraction via `raw_decode`:**
```python
obj, _ = json.JSONDecoder().raw_decode(raw)
result = DispatcherOutput.model_validate(obj)
```
`raw_decode` extracts the first valid JSON object from the model's output string, handling cases where the model emits extra tokens or multiple JSON objects separated by newlines.

**Layer 2 — Pydantic validation fallback:**
```python
except Exception:
    result = _FALLBACK  # DispatcherOutput() — all defaults
```
If any parsing or validation step fails, the pipeline falls back to a safe default `DispatcherOutput` (persona=neutral, attack=False, no predicates). This means the pipeline always continues to the Librarian stage, which will perform an unfiltered semantic search. **The pipeline never returns a 500 error due to model output parsing failure.**

**Note on vLLM guided_json:** The system is designed to optionally use vLLM's Outlines-based constrained decoding (`guided_json`) to guarantee schema compliance at the token level. The current implementation uses the post-hoc extraction approach described above, which is simpler and sufficient for the fine-tuned model that reliably produces valid JSON.

### Concierge: JSON Mode + Key Extraction

The Concierge uses OpenAI-compatible `response_format={"type": "json_object"}` to instruct vLLM to enforce valid JSON output:

```python
response = await client.chat.completions.create(
    ...
    response_format={"type": "json_object"},
)
content = response.choices[0].message.content
parsed = json.loads(content)
state.suggestion = parsed.get("suggestion", "")
state.elaboration = parsed.get("elaboration", "")
```

**No-results guard:** If the model returns the no-results sentinel phrase, any elaboration generated after it is suppressed:
```python
if state.suggestion.startswith(NO_RESULTS_RESPONSE[:40]):
    state.suggestion = NO_RESULTS_RESPONSE
    state.elaboration = ""
```

---

## pgvector Hybrid Search

The Librarian executes a single SQL query combining metadata filtering and approximate nearest-neighbor (ANN) vector search.

### Query Structure

```sql
SELECT id, name, cuisine, price_range, tier, description,
       embedding <=> CAST(:embedding AS vector) AS distance
FROM restaurants
WHERE cuisine ILIKE :cuisine          -- metadata filter (optional)
  AND price_range <= :max_price       -- metadata filter (optional)
  AND tier >= :min_tier               -- quality tier filter
ORDER BY distance                     -- ANN sort by cosine distance
LIMIT :top_k                          -- default: 5
```

### Operator: `<=>` (Cosine Distance)

The `<=>` operator computes cosine distance (1 − cosine similarity) between the query embedding and stored restaurant embeddings. Lower distance = more semantically similar. This is appropriate for sentence embeddings where the magnitude of the vector is not meaningful — only the direction (angle) matters.

### Index: IVFFlat

The `restaurants.embedding` column uses a pgvector **IVFFlat** index for ANN search:
```sql
-- From scripts/db_init.sql
CREATE INDEX ON restaurants USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);
```

IVFFlat partitions vectors into `lists` clusters (inverted file index). At query time, pgvector searches only the nearest clusters, reducing the number of distance computations from O(n) to O(n/lists). With 200 restaurants and 50 lists, each cluster holds ~4 records — effectively exact search at this dataset size.

### Pre-filter Behavior

Metadata filters are applied **before** the ANN sort in the ORDER BY clause (not as a post-filter). This means:
- The ANN search operates on the filtered subset of rows
- With a large cuisine filter (e.g., `cuisine ILIKE '%italian%'`), the effective candidate set may be small (10–30 rows), making vector similarity the tiebreaker
- The default `tier >= 2` guard ensures tier-1 restaurants never surface unless explicitly requested

### Embedding Model: all-MiniLM-L6-v2

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | **384** |
| Max sequence length | 256 tokens |
| Framework | sentence-transformers |
| Inference | CPU (embedding is fast; GPU reserved for vLLM) |

**Why all-MiniLM-L6-v2?**
- Widely used, well-benchmarked semantic similarity model
- 384-dimensional embeddings are compact (small pgvector storage overhead)
- Fast CPU inference (~1–5 ms per query) — no GPU needed for the embedding step
- Pre-trained specifically for semantic similarity tasks (trained on MS MARCO, NLI, etc.)

The same model must be used for both ingestion (`scripts/ingest.py`) and query-time embedding (`app/services/librarian.py`). The embedding dimension (384) is configured in `app/config.py` as `embedding_dim: int = 384` and used in the `db_init.sql` schema: `embedding VECTOR(384)`.

---

## Full Request Latency Profile (Approximate)

On RTX 3060 with 4-bit quantization:

| Stage | Component | Estimated Latency |
|-------|-----------|------------------|
| Stage 1 | Dispatcher inference (1B model, ~50–80 output tokens) | 200–500 ms |
| Stage 2 | Embedding (all-MiniLM-L6-v2, CPU) + pgvector query | 10–30 ms |
| Stage 3 | Concierge inference (3B model, ~100–200 output tokens) | 500–1200 ms |
| **Total** | **End-to-end** | **~0.7–1.7 seconds** |

Attack short-circuit path (no Stage 2/3): ~200–500 ms (Dispatcher only).
