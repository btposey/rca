#!/usr/bin/env bash
# check_deps.sh — verify GPU machine dependencies before pod launch
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    local label="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}[PASS]${NC} $label"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}[FAIL]${NC} $label"
        FAIL=$((FAIL + 1))
    fi
}

warn() {
    local label="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "${GREEN}[PASS]${NC} $label"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}[WARN]${NC} $label (non-fatal)"
    fi
}

echo "=== RCA Dependency Check ==="
echo ""

# ── Core tools ───────────────────────────────────────────────────────────────
check "curl installed"             "command -v curl"
check "git installed"              "command -v git"
warn  "uv installed"               "command -v uv || test -x $HOME/.local/bin/uv"
check "rsync installed"            "command -v rsync"

# ── Podman ───────────────────────────────────────────────────────────────────
check "podman installed"           "command -v podman"
check "podman version >= 4"        "podman --version | grep -oP '[\d]+' | head -1 | xargs -I{} test {} -ge 4"

# ── NVIDIA / CUDA ─────────────────────────────────────────────────────────────
check "nvidia-smi available"       "command -v nvidia-smi"
check "GPU detected"               "nvidia-smi --query-gpu=name --format=csv,noheader | grep -q ."
check "CUDA libraries present"     "find /usr/lib -name 'libcuda.so*' 2>/dev/null | grep -q ."
check "nvidia-ctk installed"       "command -v nvidia-ctk"
check "nvidia CDI configured"      "nvidia-ctk cdi list 2>/dev/null | grep -q nvidia"

# ── Disk space (>= 30GB free in project parent) ───────────────────────────────
check ">=30GB free in /home"       "df /home --output=avail | tail -1 | awk '{exit (\$1/1024/1024 < 30)}'"

# ── Model path ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${MODEL_HOST_PATH:-${PROJECT_ROOT}/models}"
warn "Model path exists ($MODEL_PATH)"  "test -d '$MODEL_PATH'"

# ── Ports ─────────────────────────────────────────────────────────────────────
check "port 8000 available"        "! ss -tlnp | grep -q ':8000 '"
check "port 8001 available"        "! ss -tlnp | grep -q ':8001 '"
check "port 5432 available"        "! ss -tlnp | grep -q ':5432 '"
check "port 8080 available"        "! ss -tlnp | grep -q ':8080 '"

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}Fix failures before running deploy.sh${NC}"
    echo -e "If this is a fresh machine, run: ${YELLOW}bash scripts/setup_gpu_machine.sh${NC}"
    exit 1
else
    echo -e "${GREEN}All checks passed. Ready to deploy.${NC}"
fi
