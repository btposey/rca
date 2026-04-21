#!/usr/bin/env bash
# deploy_native_vllm.sh — launch vLLM instances natively on the host alongside the pod.
#
# Use this instead of deploy.sh when podman play kube does not support CDI
# GPU device passthrough (podman < 5.x). Once podman 5.x is available,
# use deploy.sh which runs all services in the pod.
#
# Topology:
#   - vLLM Dispatcher  : native process, port 8000
#   - vLLM Concierge   : native process, port 8001
#   - Postgres + API   : podman pod (rca-pod), ports 5432 + 8080
#
# Run from the rca/ project root on the GPU machine:
#   bash scripts/deploy_native_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/deploy/pod.env"
export PATH="$HOME/.local/bin:$PROJECT_ROOT/.venv/bin:$PATH"
VLLM="$PROJECT_ROOT/.venv/bin/vllm"

MODEL_HOST_PATH="${MODEL_HOST_PATH:-${PROJECT_ROOT}/models}"
POSTGRES_DATA_PATH="${POSTGRES_DATA_PATH:-${PROJECT_ROOT}/data/postgres}"
LOG_DIR="${PROJECT_ROOT}/logs"

echo "=== RCA Deploy (native vLLM) ==="

# ---------------------------------------------------------------------------
# Helper: kill processes bound to a TCP port without touching the SSH session.
# Uses ss to find the PID, then verifies it is a vllm process before killing.
# ---------------------------------------------------------------------------
safe_kill_port() {
    local port="$1"
    # ss output for a listening socket looks like:
    #   LISTEN 0 128 0.0.0.0:<port> ... users:(("vllm",pid=12345,fd=7))
    local pids
    pids=$(ss -tlnp "sport = :${port}" 2>/dev/null \
           | grep -oP '(?<=pid=)\d+' || true)

    if [[ -z "$pids" ]]; then
        return 0
    fi

    for pid in $pids; do
        # Only kill if the process name contains "vllm" (or "python" running vllm).
        local comm
        comm=$(ps -p "$pid" -o comm= 2>/dev/null || true)
        local cmdline
        cmdline=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null || true)
        if echo "$comm $cmdline" | grep -qi "vllm"; then
            echo "  Killing vllm PID $pid on port $port (${comm})"
            kill "$pid" 2>/dev/null || true
        else
            echo "  WARNING: PID $pid on port $port does not appear to be vllm (${comm}) — skipping kill"
        fi
    done
}

# ---------------------------------------------------------------------------
# Helper: wait for a vLLM HTTP health endpoint to become ready.
# Usage: wait_for_vllm_health <url> <label> <max_seconds> <interval_seconds>
# ---------------------------------------------------------------------------
wait_for_vllm_health() {
    local url="$1"
    local label="$2"
    local max_wait="${3:-120}"
    local interval="${4:-3}"
    local elapsed=0

    echo "  Waiting for ${label} at ${url} (timeout ${max_wait}s) ..."
    while true; do
        if curl -sf --max-time 2 "${url}" > /dev/null 2>&1; then
            echo "  [OK] ${label} is healthy (${elapsed}s)"
            return 0
        fi

        if (( elapsed >= max_wait )); then
            echo "  [ERROR] ${label} did not become healthy within ${max_wait}s" >&2
            echo "          Check logs for details:" >&2
            return 1
        fi

        sleep "$interval"
        elapsed=$(( elapsed + interval ))
        echo "  ... still waiting for ${label} (${elapsed}s / ${max_wait}s)"
    done
}

# 1. Stop existing services first so port checks pass
echo "--- Stopping existing services ---"
if podman pod exists "$POD_NAME" 2>/dev/null; then
    podman pod stop "$POD_NAME" || true
    podman pod rm "$POD_NAME" || true
fi
safe_kill_port "${VLLM_DISPATCHER_PORT}"
safe_kill_port "${VLLM_CONCIERGE_PORT}"
sleep 2

# 2. Pre-flight
bash "$SCRIPT_DIR/check_deps.sh"

# 3. Ensure directories exist
mkdir -p "$MODEL_HOST_PATH" "$LOG_DIR"

# Validate postgres data directory — if it exists but is owned by a different
# UID (stale from a previous rootless container run), create a fresh one.
if [[ -d "$POSTGRES_DATA_PATH" ]] && [[ -n "$(ls -A "$POSTGRES_DATA_PATH" 2>/dev/null)" ]]; then
    if ! touch "$POSTGRES_DATA_PATH/.write_test" 2>/dev/null; then
        echo "[WARN] $POSTGRES_DATA_PATH has permission issues (stale container data)."
        POSTGRES_DATA_PATH="${POSTGRES_DATA_PATH%/}-$(date +%s)"
        echo "       Using fresh directory: $POSTGRES_DATA_PATH"
    else
        rm -f "$POSTGRES_DATA_PATH/.write_test"
    fi
fi
mkdir -p "$POSTGRES_DATA_PATH"

# 3b. Download fine-tuned models from HuggingFace if not present locally
HF_DISPATCHER="${MODEL_HOST_PATH}/dispatcher-llama-1b"
HF_CONCIERGE="${MODEL_HOST_PATH}/concierge-llama-3b"
HF_TOKEN_FILE="${PROJECT_ROOT}/HF_TOKEN"
HF_TOKEN_VAL="${HF_TOKEN:-}"

if [[ -z "$HF_TOKEN_VAL" && -f "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN_VAL=$(grep -oP '(?<=HF_TOKEN=).+' "$HF_TOKEN_FILE" || cat "$HF_TOKEN_FILE")
fi

download_model_if_missing() {
    local local_path="$1"
    local repo_id="$2"
    if [[ ! -f "${local_path}/config.json" ]]; then
        echo "  Model not found at ${local_path} — downloading from HuggingFace..."
        if [[ -n "$HF_TOKEN_VAL" ]]; then
            "$PROJECT_ROOT/.venv/bin/python3" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${repo_id}', local_dir='${local_path}', token='${HF_TOKEN_VAL}')
print('Downloaded ${repo_id}')
"
        else
            echo "  [WARN] No HF_TOKEN found — attempting public download of ${repo_id}"
            "$PROJECT_ROOT/.venv/bin/python3" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='${repo_id}', local_dir='${local_path}')
print('Downloaded ${repo_id}')
"
        fi
    else
        echo "  Model present: ${local_path}"
    fi
}

echo "--- Checking model weights ---"
download_model_if_missing "$HF_DISPATCHER" "bposey-flexion/rca-dispatcher-llama-1b"
download_model_if_missing "$HF_CONCIERGE"  "bposey-flexion/rca-concierge-llama-3b"

# 3a. Rotate existing logs so this deploy starts clean
echo "--- Rotating logs ---"
for logfile in dispatcher concierge; do
    src="${LOG_DIR}/${logfile}.log"
    dst="${LOG_DIR}/${logfile}.log.prev"
    if [[ -f "$src" ]]; then
        mv -f "$src" "$dst"
        echo "  Rotated ${logfile}.log -> ${logfile}.log.prev"
    fi
done

# 4. Build images
echo ""
echo "--- Building API image ---"
podman build -f "$PROJECT_ROOT/Dockerfile" -t rca-api:latest "$PROJECT_ROOT"

echo ""
echo "--- Building UI image ---"
podman build -f "$PROJECT_ROOT/Dockerfile.ui" -t rca-ui:latest --no-cache "$PROJECT_ROOT"

# 5. Launch vLLM Dispatcher natively
echo ""
echo "--- Starting Dispatcher (port ${VLLM_DISPATCHER_PORT}) ---"
nohup "$VLLM" serve "${MODEL_HOST_PATH}/dispatcher-llama-1b" \
    --served-model-name dispatcher-llama-1b \
    --host 0.0.0.0 \
    --port "${VLLM_DISPATCHER_PORT}" \
    --quantization bitsandbytes \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.40 \
    --max-num-seqs 8 \
    > "${LOG_DIR}/dispatcher.log" 2>&1 &
DISPATCHER_PID=$!
echo "  PID ${DISPATCHER_PID} — logs: ${LOG_DIR}/dispatcher.log"

# 6. Launch vLLM Concierge natively
echo "--- Starting Concierge (port ${VLLM_CONCIERGE_PORT}) ---"
nohup "$VLLM" serve "${MODEL_HOST_PATH}/concierge-llama-3b" \
    --served-model-name concierge-llama-3b \
    --host 0.0.0.0 \
    --port "${VLLM_CONCIERGE_PORT}" \
    --quantization bitsandbytes \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.50 \
    > "${LOG_DIR}/concierge.log" 2>&1 &
CONCIERGE_PID=$!
echo "  PID ${CONCIERGE_PID} — logs: ${LOG_DIR}/concierge.log"

# 6a. Wait for both vLLM processes to be healthy before proceeding
echo ""
echo "--- Waiting for vLLM health ---"
if ! wait_for_vllm_health \
        "http://localhost:${VLLM_DISPATCHER_PORT}/health" \
        "Dispatcher (port ${VLLM_DISPATCHER_PORT})" 120 3; then
    echo "Dispatcher failed to start. Last 20 log lines:" >&2
    tail -20 "${LOG_DIR}/dispatcher.log" >&2 || true
    exit 1
fi

if ! wait_for_vllm_health \
        "http://localhost:${VLLM_CONCIERGE_PORT}/health" \
        "Concierge (port ${VLLM_CONCIERGE_PORT})" 120 3; then
    echo "Concierge failed to start. Last 20 log lines:" >&2
    tail -20 "${LOG_DIR}/concierge.log" >&2 || true
    exit 1
fi

# 7. Launch pod (postgres + api + ui)
echo ""
echo "--- Launching rca-pod (postgres + api + ui) ---"

# Detect host IP — used by containers to reach vLLM and postgres on the host
HOST_IP="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
echo "  Host IP: ${HOST_IP}"

RESOLVED_YAML="/tmp/rca-pod-resolved.yaml"
sed \
    -e "s|MODEL_HOST_PATH_PLACEHOLDER|${MODEL_HOST_PATH}|g" \
    -e "s|POSTGRES_DATA_PATH_PLACEHOLDER|${POSTGRES_DATA_PATH}|g" \
    -e "s|HOST_IP_PLACEHOLDER|${HOST_IP}|g" \
    "$PROJECT_ROOT/deploy/pod.yaml" > "$RESOLVED_YAML"
podman play kube --network host "$RESOLVED_YAML"

# 8. Initialize database and load restaurant data if table is empty
echo ""
echo "--- Initializing database ---"
sleep 10  # wait for postgres to be ready
podman exec -i rca-pod-postgres psql -U rca -d rca < "$PROJECT_ROOT/scripts/db_init.sql" 2>/dev/null || true

RESTAURANT_COUNT=$(podman exec rca-pod-postgres psql -U rca -d rca -tAc "SELECT COUNT(*) FROM restaurants;" 2>/dev/null || echo "0")
if [[ "$RESTAURANT_COUNT" -eq "0" ]]; then
    echo "  Loading restaurant data..."
    SEED_FILE="$PROJECT_ROOT/data/restaurant_data/synthetic_restaurants.json"
    if [[ -f "$SEED_FILE" ]]; then
        "$PROJECT_ROOT/.venv/bin/python3" -m scripts.ingest --source "$SEED_FILE" 2>&1 | tail -3
        echo "  Restaurant data loaded."
    else
        echo "  [WARN] No restaurant data found at $SEED_FILE"
    fi
else
    echo "  Database already has ${RESTAURANT_COUNT} restaurants — skipping ingest."
fi

# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                     RCA Stack — Startup Summary                     ║"
echo "╠══════════════════════════════════════╦══════════════════════════════╣"
printf "║  %-36s║  %-28s║\n" "Service"              "URL / Location"
echo "╠══════════════════════════════════════╬══════════════════════════════╣"
printf "║  %-36s║  %-28s║\n" "UI (Streamlit)"        "http://localhost:8501"
printf "║  %-36s║  %-28s║\n" "API"                   "http://localhost:${API_PORT}"
printf "║  %-36s║  %-28s║\n" "Dispatcher vLLM (PID ${DISPATCHER_PID})" \
                               "http://localhost:${VLLM_DISPATCHER_PORT}/v1"
printf "║  %-36s║  %-28s║\n" "Concierge  vLLM (PID ${CONCIERGE_PID})" \
                               "http://localhost:${VLLM_CONCIERGE_PORT}/v1"
printf "║  %-36s║  %-28s║\n" "Postgres"              "localhost:${POSTGRES_PORT}"
echo "╠══════════════════════════════════════╩══════════════════════════════╣"
echo "║  vLLM Health                                                        ║"
printf "║    Dispatcher : http://localhost:%-4s/health  [OK]               ║\n" "${VLLM_DISPATCHER_PORT}"
printf "║    Concierge  : http://localhost:%-4s/health  [OK]               ║\n" "${VLLM_CONCIERGE_PORT}"
echo "╠═════════════════════════════════════════════════════════════════════╣"
echo "║  Log files                                                          ║"
printf "║    tail -f %-57s║\n" "${LOG_DIR}/dispatcher.log"
printf "║    tail -f %-57s║\n" "${LOG_DIR}/concierge.log"
echo "║  Pod logs                                                           ║"
echo "║    podman logs rca-pod-api                                          ║"
echo "║    podman logs rca-pod-postgres                                     ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
