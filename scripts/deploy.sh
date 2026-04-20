#!/usr/bin/env bash
# deploy.sh — build and launch the rca-pod on the GPU machine
# Run from the rca/ project root on the GPU machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_ROOT/deploy/pod.env"

# Ensure uv is on PATH if installed via installer script
export PATH="$HOME/.local/bin:$PATH"

echo "=== RCA Deploy ==="
echo "Pod:        $POD_NAME"
echo "Models:     $MODEL_HOST_PATH"
echo "API port:   $API_PORT"

# 1. Pre-flight checks
bash "$SCRIPT_DIR/check_deps.sh"

# 2. Ensure host paths exist
mkdir -p "$MODEL_HOST_PATH"
mkdir -p "$POSTGRES_DATA_PATH"

# 3. Build vLLM image (shared by dispatcher + concierge containers)
echo ""
echo "--- Building vLLM image ---"
podman build -f "$PROJECT_ROOT/Dockerfile.vllm" -t rca-vllm:latest "$PROJECT_ROOT"

# 4. Build API image
echo ""
echo "--- Building API image ---"
podman build -f "$PROJECT_ROOT/Dockerfile" -t rca-api:latest "$PROJECT_ROOT"

# 5. Substitute path placeholders in pod_full.yaml
RESOLVED_YAML="/tmp/rca-pod-resolved.yaml"
sed \
  -e "s|MODEL_HOST_PATH_PLACEHOLDER|${MODEL_HOST_PATH}|g" \
  -e "s|POSTGRES_DATA_PATH_PLACEHOLDER|${POSTGRES_DATA_PATH}|g" \
  "$PROJECT_ROOT/deploy/pod_full.yaml" > "$RESOLVED_YAML"

# 6. Stop existing pod if running
if podman pod exists "$POD_NAME" 2>/dev/null; then
    echo "--- Stopping existing pod ---"
    podman pod stop "$POD_NAME" || true
    podman pod rm "$POD_NAME" || true
fi

# 7. Launch pod
echo ""
echo "--- Launching pod ---"
podman play kube "$RESOLVED_YAML"

echo ""
echo "=== Pod launched ==="
echo "API:             http://localhost:${API_PORT}"
echo "Dispatcher:      http://localhost:${VLLM_DISPATCHER_PORT}/v1"
echo "Concierge:       http://localhost:${VLLM_CONCIERGE_PORT}/v1"
echo "Postgres:        localhost:${POSTGRES_PORT}"
echo ""
echo "Check status: podman pod ps"
echo "View logs:    podman logs rca-pod-api"
