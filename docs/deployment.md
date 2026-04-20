# Deployment Guide

## Overview

RCA is deployed on a single GPU machine running Linux with an NVIDIA RTX 3060 (12 GB VRAM). The production topology separates vLLM inference (native host processes) from the API and database (Podman pod).

**Target machine:** `192.168.200.100` — Linux, RTX 3060
**Dev machine:** macOS — source code only; no services run natively on Mac

---

## Architecture: Native vLLM + Podman Pod

Due to a GPU passthrough limitation in Podman 4.x (see rationale below), the deployed topology splits services across two layers:

```
Host (GPU machine)
├── vLLM Dispatcher     (native Python process, port 8000)
│   └── Llama 3.2 1B fine-tuned, 4-bit bitsandbytes, GPU
├── vLLM Concierge      (native Python process, port 8001)
│   └── Llama 3.2 3B, 4-bit bitsandbytes, GPU
│
└── rca-pod (Podman — host network mode)
    ├── rca-pod-postgres  (pgvector/pgvector:pg16, port 5432)
    │   └── data volume: $POSTGRES_DATA_PATH
    └── rca-pod-api       (rca-api:latest, port 8080)
        └── FastAPI app — calls vLLM on localhost:8000/8001
```

All containers in the pod use `--network host`, so they share the host's network namespace. The API container reaches vLLM on `localhost:8000` and `localhost:8001`, and PostgreSQL on `localhost:5432`.

---

## vLLM Native Hosting Rationale

**Problem:** Podman 4.x does not support CDI (Container Device Interface) GPU device passthrough when using `podman play kube` (k8s-style pod YAML). CDI support for `play kube` requires Podman 5.x, which is not available on the target machine.

**Solution:** Run vLLM as native host processes using the project's virtualenv, launched with `nohup` and redirected to log files. The Podman pod handles only stateless (API) and stateful-but-CPU (PostgreSQL) services that do not require GPU access.

This is the documented workaround in `scripts/deploy_native_vllm.sh`. When Podman 5.x becomes available, the full-pod deployment (`scripts/deploy.sh` + `deploy/pod_full.yaml`) can be used instead.

---

## Prerequisites

### 1. Check Dependencies (`scripts/check_deps.sh`)

Run before any deployment to verify all requirements:

```bash
bash scripts/check_deps.sh
```

Checks performed:

| Check | Requirement |
|-------|-------------|
| curl, git, rsync | Core tools present |
| uv | Python package manager (warn-only if missing) |
| podman | Version >= 4 |
| nvidia-smi | GPU detected and queryable |
| CUDA libraries | `/usr/lib/libcuda.so*` present |
| nvidia-ctk | NVIDIA Container Toolkit installed |
| CDI config | `nvidia-ctk cdi list` returns nvidia entries |
| Disk space | >= 30 GB free in `/home` |
| Model path | `$MODEL_HOST_PATH` directory exists (warn-only) |
| Ports | 8000, 8001, 5432, 8080 all available |

If any required check fails, the script exits with code 1 and prints remediation instructions. On a fresh machine, run `scripts/setup_gpu_machine.sh` first.

### 2. First-Time Machine Setup (`scripts/setup_gpu_machine.sh`)

Installs and configures:
- NVIDIA drivers and CUDA toolkit
- Podman and nvidia-ctk
- uv Python package manager
- Required host directories (`$MODEL_HOST_PATH`, `$POSTGRES_DATA_PATH`)

### 3. Model Weights

Fine-tuned model weights must be present at `$MODEL_HOST_PATH` before deployment:

```
$MODEL_HOST_PATH/
├── dispatcher-llama-1b/    # merged QLoRA fine-tune of Llama 3.2 1B
└── concierge-llama-3b/     # Llama 3.2 3B Instruct (downloaded from HuggingFace)
```

Models are downloaded from HuggingFace during training (`HUGGING_FACE_HUB_TOKEN` required) or can be pre-staged manually.

---

## Local Deployment (Primary Path)

### Step 1: Configure Environment

```bash
cp .env.example .env
# Edit .env — required fields:
#   HUGGING_FACE_HUB_TOKEN=<your token>
#   ANTHROPIC_API_KEY=<your key>   # only needed for dataset generation
```

### Step 2: Install Python Dependencies

```bash
# On GPU machine, in the rca/ project root:
uv sync
```

### Step 3: Deploy (Native vLLM)

```bash
bash scripts/deploy_native_vllm.sh
```

This script:
1. Stops any existing `rca-pod` and kills processes on ports 8000/8001
2. Runs `check_deps.sh` pre-flight checks
3. Creates required host directories
4. Builds the API container image (`rca-api:latest`) and UI image (`rca-ui:latest`)
5. Launches vLLM Dispatcher natively (port 8000, `nohup`, logs → `logs/dispatcher.log`)
6. Launches vLLM Concierge natively (port 8001, `nohup`, logs → `logs/concierge.log`)
7. Resolves path placeholders in `deploy/pod.yaml` and launches the Podman pod (postgres + api)

### Step 4: Initialize Database

Wait ~30 seconds for PostgreSQL to be ready, then:

```bash
# Create schema + pgvector extension
podman exec rca-pod-postgres psql -U rca -d rca -f /scripts/db_init.sql

# Ingest restaurant data (embeds + loads 200 restaurant records)
uv run scripts/ingest.py --source data/restaurant_data/synthetic_restaurants.json
```

### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8080/health
# → {"status": "ok"}

# Test query
curl -s -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "I want a proper omakase, money is no object"}' | jq .
```

### Log Monitoring

```bash
# vLLM logs (native processes)
tail -f logs/dispatcher.log
tail -f logs/concierge.log

# Pod container logs
podman logs rca-pod-api
podman logs rca-pod-postgres

# Pod status
podman pod ps
podman ps --pod
```

---

## Pod Architecture Details

### Pod Definition

The Podman pod is defined in `deploy/pod.yaml` (lightweight: postgres + api only) and `deploy/pod_full.yaml` (full: postgres + api + both vLLM containers, for use when Podman 5.x CDI is available).

The YAML contains two placeholder strings that are substituted at deploy time:
- `MODEL_HOST_PATH_PLACEHOLDER` → actual model path from `pod.env`
- `POSTGRES_DATA_PATH_PLACEHOLDER` → actual postgres data path from `pod.env`

### Pod Environment (`deploy/pod.env`)

```bash
MODEL_HOST_PATH=/home/brian/Workspaces/ai-class/final/models
POSTGRES_DATA_PATH=/home/brian/Workspaces/ai-class/final/postgres-data
POD_NAME=rca-pod
API_PORT=8080
VLLM_DISPATCHER_PORT=8000
VLLM_CONCIERGE_PORT=8001
POSTGRES_PORT=5432
```

### Containers

| Container | Image | Port | Role |
|-----------|-------|------|------|
| `rca-pod-postgres` | `docker.io/pgvector/pgvector:pg16` | 5432 | PostgreSQL 16 with pgvector extension |
| `rca-pod-api` | `rca-api:latest` (built locally) | 8080 | FastAPI application |

The UI container (`rca-ui:latest`) is built but runs separately from the pod in some configurations (port 8501).

---

## Environment Variables

All variables are documented in `.env.example`. At runtime, `app/config.py` loads them via `pydantic-settings`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | `` | Anthropic API key (dataset generation only) |
| `DATABASE_URL` | `postgresql+asyncpg://rca:rca@localhost:5432/rca` | PostgreSQL connection string (asyncpg driver) |
| `VLLM_DISPATCHER_BASE_URL` | `http://localhost:8000/v1` | Dispatcher vLLM endpoint |
| `VLLM_DISPATCHER_MODEL` | `dispatcher-llama-1b` | Served model name for Dispatcher |
| `VLLM_CONCIERGE_BASE_URL` | `http://localhost:8001/v1` | Concierge vLLM endpoint |
| `VLLM_CONCIERGE_MODEL` | `concierge-llama-3b` | Served model name for Concierge |
| `HF_HOME` | `/models` | HuggingFace model cache directory |
| `HUGGING_FACE_HUB_TOKEN` | `` | HuggingFace access token (gated model download) |
| `DISPATCHER_TEMPERATURE` | `0.1` | Override Dispatcher sampling temperature |
| `DISPATCHER_MAX_TOKENS` | `256` | Override Dispatcher max output tokens |
| `CONCIERGE_TEMPERATURE` | `0.7` | Override Concierge sampling temperature |
| `CONCIERGE_TOP_P` | `0.9` | Override Concierge nucleus sampling threshold |
| `CONCIERGE_MAX_TOKENS` | `512` | Override Concierge max output tokens |
| `TOP_K_RESULTS` | `5` | Number of restaurant candidates to retrieve |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for Librarian |

---

## Training Workflow (GPU Machine)

Before deployment, models must be trained. The orchestration scripts handle the full pipeline:

```bash
# Generate training data (run on Mac using AWS Bedrock — no GPU needed)
uv run scripts/generate_dataset_bedrock.py

# Rsync data to GPU machine, fine-tune, and evaluate
uv run scripts/train_dispatcher.py
uv run scripts/train_concierge.py  # optional — only if fine-tuning Concierge
```

`train_dispatcher.py` orchestrates:
1. Generate Dispatcher training data via Claude/Bedrock (teacher model)
2. Rsync data files to GPU machine
3. SSH to GPU machine and run `finetune.py` (QLoRA via Unsloth)
4. Run `evaluate.py` against held-out eval set
5. Report metrics

---

## GCP Deployment (Deferred)

GCP deployment is deferred and not required for the project deadline.

**Planned approach:**
- Cloud Run or GKE Autopilot for the FastAPI API container
- Cloud SQL (PostgreSQL) with pgvector extension for the database
- A100/T4 GPU instance for vLLM serving
- GitHub Actions workflow (`.github/workflows/deploy-gcp.yml`) for CI/CD

The GitHub Actions workflow directory (`.github/workflows/`) is scaffolded but no deploy-gcp.yml has been created yet. GCP deployment would require:
1. A GCP project with Vertex AI or Compute Engine GPU quota
2. Cloud SQL instance with pgvector enabled
3. Artifact Registry for container images
4. Service account credentials in GitHub Secrets

**Why deferred:** The RTX 3060 on-premises machine meets all requirements for the project submission. GCP introduces cost and complexity without adding correctness or rubric points.

---

## Deployment Checklist

- [ ] `bash scripts/check_deps.sh` — all checks pass
- [ ] `.env` configured (HF token, API key)
- [ ] Model weights present at `$MODEL_HOST_PATH/dispatcher-llama-1b/` and `$MODEL_HOST_PATH/concierge-llama-3b/`
- [ ] `bash scripts/deploy_native_vllm.sh` — completes without error
- [ ] PostgreSQL schema initialized (`db_init.sql`)
- [ ] Restaurant data ingested (`scripts/ingest.py`)
- [ ] `GET /health` returns `{"status": "ok"}`
- [ ] `POST /query` with test query returns valid response
