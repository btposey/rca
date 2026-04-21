#!/usr/bin/env bash
# setup_gpu_machine.sh — one-time setup for the RCA GPU machine.
#
# Installs:
#   - NVIDIA Container Toolkit (nvidia-ctk) for GPU access in containers
#   - Configures CDI so podman can assign GPUs via pod.yaml resource limits
#   - Creates required host directories
#
# Run once on the GPU machine before deploy.sh:
#   bash scripts/setup_gpu_machine.sh
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
require() { command -v "$1" &>/dev/null || { echo -e "${RED}[FAIL]${NC} Required command not found: $1"; exit 1; }; }

echo "=== RCA GPU Machine Setup ==="
echo ""

# ── Prerequisites ────────────────────────────────────────────────────────────
require nvidia-smi
require sudo

info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ── System packages ───────────────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    curl git rsync podman
info "System packages installed."

# ── uv ────────────────────────────────────────────────────────────────────────
if command -v uv &>/dev/null; then
    info "uv already installed: $(uv --version)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    info "uv installed."
fi

# ── NVIDIA Container Toolkit ─────────────────────────────────────────────────
if command -v nvidia-ctk &>/dev/null; then
    info "nvidia-ctk already installed: $(nvidia-ctk --version 2>&1 | head -1)"
else
    info "Installing NVIDIA Container Toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
    info "nvidia-ctk installed."
fi

# ── CDI Configuration ─────────────────────────────────────────────────────────
CDI_FILE="/etc/cdi/nvidia.yaml"
if nvidia-ctk cdi list 2>/dev/null | grep -q nvidia; then
    info "CDI already configured."
else
    info "Generating CDI configuration..."
    sudo mkdir -p /etc/cdi
    sudo nvidia-ctk cdi generate --output="$CDI_FILE"
    info "CDI configured: $CDI_FILE"
fi

info "CDI devices:"
nvidia-ctk cdi list 2>/dev/null | sed 's/^/  /'

# ── Host directories ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${MODEL_HOST_PATH:-${PROJECT_ROOT}/models}"
POSTGRES_PATH="${POSTGRES_DATA_PATH:-${PROJECT_ROOT}/data/postgres}"

info "Creating host directories..."
mkdir -p "$MODEL_PATH"
mkdir -p "$POSTGRES_PATH"
info "  $MODEL_PATH"
info "  $POSTGRES_PATH"

echo ""
echo "=== Setup complete. Run deploy.sh to launch the pod. ==="
