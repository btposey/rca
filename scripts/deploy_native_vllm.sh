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

MODEL_HOST_PATH="${MODEL_HOST_PATH:-/home/brian/Workspaces/ai-class/final/models}"
POSTGRES_DATA_PATH="${POSTGRES_DATA_PATH:-/home/brian/Workspaces/ai-class/final/postgres-data}"
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
mkdir -p "$MODEL_HOST_PATH" "$POSTGRES_DATA_PATH" "$LOG_DIR"

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
    --max-model-len 2048 \
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

# 7. Launch pod (postgres + api only)
echo ""
echo "--- Launching rca-pod (postgres + api) ---"
RESOLVED_YAML="/tmp/rca-pod-resolved.yaml"
sed \
    -e "s|MODEL_HOST_PATH_PLACEHOLDER|${MODEL_HOST_PATH}|g" \
    -e "s|POSTGRES_DATA_PATH_PLACEHOLDER|${POSTGRES_DATA_PATH}|g" \
    "$PROJECT_ROOT/deploy/pod.yaml" > "$RESOLVED_YAML"
podman play kube --network host "$RESOLVED_YAML"

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
