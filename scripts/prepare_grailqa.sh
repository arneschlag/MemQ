#!/usr/bin/env bash
# Prepare one labelled GrailQA-style development split for zero-shot v9 inference.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-grailqa}"
INPUT="${2:-}"
case "$DATASET" in
  grailqa)
    INPUT="${INPUT:-$ROOT/data/grailqa/grailqa_v1.0_dev.json}"
    ;;
  grailqa++)
    INPUT="${INPUT:-$ROOT/data/grailqa++/grailqa++_dev.json}"
    ;;
  *) echo "Usage: $0 [grailqa|grailqa++] [labelled-dev.json]" >&2; exit 2 ;;
esac
[[ -s "$INPUT" ]] || { echo "Missing labelled dataset: $INPUT" >&2; exit 1; }
cd "$ROOT"
PYTHON="${MEMQ_PYTHON:-python}"
[[ -x .venv/bin/python ]] && PYTHON=".venv/bin/python"
"$PYTHON" grailqa_adapter.py --dataset "$DATASET" --input "$INPUT"
