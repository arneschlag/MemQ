#!/usr/bin/env bash
# Guided MemQ setup. Run without arguments for interactive setup, or use:
# scripts/setup.sh [auto|cpu|cuda|rocm] [--weights] [--raw-data] [--non-interactive]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=""
GET_WEIGHTS=0
GET_RAW_DATA=0
NON_INTERACTIVE=0
ENDPOINT="${MEMQ_SPARQL_ENDPOINT:-}"

for arg in "$@"; do
  case "$arg" in
    auto|cpu|cuda|rocm) MODE="$arg" ;;
    --weights) GET_WEIGHTS=1 ;;
    --raw-data) GET_RAW_DATA=1 ;;
    --non-interactive) NON_INTERACTIVE=1 ;;
    -h|--help) sed -n '1,3p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ask_yes_no() {
  local prompt="$1" answer
  read -r -p "$prompt [y/N] " answer
  [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]
}

if [[ -z "$MODE" && "$NON_INTERACTIVE" == 0 && -t 0 ]]; then
  echo "MemQ setup"
  echo "Choose the PyTorch platform:"
  echo "  1) CPU only (lookup/evaluation; no GPU required)"
  echo "  2) NVIDIA GPU (CUDA)"
  echo "  3) AMD GPU (ROCm)"
  echo "  4) Detect automatically"
  read -r -p "Selection [1-4, default 4]: " choice
  case "${choice:-4}" in
    1) MODE="cpu" ;;
    2) MODE="cuda" ;;
    3) MODE="rocm" ;;
    4) MODE="auto" ;;
    *) echo "Please choose 1, 2, 3, or 4." >&2; exit 2 ;;
  esac
  echo "Answer-level metrics need a Freebase/Virtuoso SPARQL endpoint."
  read -r -p "Endpoint URL (or 'skip') [http://localhost:3001/sparql]: " input_endpoint
  ENDPOINT="${input_endpoint:-http://localhost:3001/sparql}"
  if ask_yes_no "Download the 14.97 GiB Llama-3-MemQ-v9 model now?"; then GET_WEIGHTS=1; fi
  if ask_yes_no "Also download the 65 MiB WebQSP/CWQ source datasets?"; then GET_RAW_DATA=1; fi
fi

MODE="${MODE:-auto}"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.9+ is required. Install Python, then run this script again." >&2
  exit 1
fi

if [[ "$MODE" == auto ]]; then
  if command -v rocm-smi >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then MODE="rocm"
  elif command -v nvidia-smi >/dev/null 2>&1; then MODE="cuda"
  else MODE="cpu"
  fi
fi

case "$MODE" in
  cpu)  TORCH_INDEX="${PYTORCH_CPU_INDEX:-https://download.pytorch.org/whl/cpu}" ;;
  cuda) TORCH_INDEX="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}" ;;
  rocm) TORCH_INDEX="${PYTORCH_ROCM_INDEX:-https://download.pytorch.org/whl/rocm6.3}" ;;
  *) echo "Internal error: unsupported mode $MODE" >&2; exit 2 ;;
esac

if [[ "$ENDPOINT" == "skip" ]]; then ENDPOINT=""
elif [[ -n "$ENDPOINT" && ( ! "$ENDPOINT" =~ ^https?:// || "$ENDPOINT" =~ [[:space:]] || "$ENDPOINT" == *"'"* || "$ENDPOINT" == *'"'* ) ]]; then
  echo "The endpoint must be a quote-free http(s) URL, or be 'skip'." >&2
  exit 2
fi

cd "$ROOT"
VENV="${MEMQ_VENV:-$ROOT/.venv}"
echo "Installing $MODE PyTorch from $TORCH_INDEX"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch --index-url "$TORCH_INDEX"
"$VENV/bin/python" -m pip install -r requirements.txt

data_args=()
if [[ "$GET_RAW_DATA" == 1 ]]; then data_args+=(--raw); fi
./scripts/download_reproduction_data.sh "${data_args[@]}"
if [[ "$GET_WEIGHTS" == 1 ]]; then ./scripts/download_weights.sh; fi

if [[ -n "$ENDPOINT" ]]; then
  cat > .env <<EOF
# Created by scripts/setup.sh; local-only and ignored by Git.
export MEMQ_SPARQL_ENDPOINT='$ENDPOINT'
EOF
  if command -v curl >/dev/null 2>&1; then
    status="$(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' "$ENDPOINT" || true)"
    if [[ "$status" =~ ^2|^3 ]]; then
      echo "Freebase endpoint check: HTTP $status"
    else
      echo "Warning: endpoint check returned '${status:-no response}'. Lookup still works; answer scoring needs a reachable endpoint." >&2
    fi
  fi
fi

"$VENV/bin/python" - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
print(f"Torch CUDA version: {torch.version.cuda}")
print(f"Torch HIP version: {torch.version.hip}")
PY

echo "Setup complete. Lookup: scripts/reproduce_lookup.sh webqsp"
[[ -n "$ENDPOINT" ]] && echo "Answer scoring: scripts/reproduce_answers.sh webqsp"
