#!/usr/bin/env bash
# Prepare one labelled GrailQA-style development split for zero-shot v9 inference.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-grailqa}"
SPLIT="${2:-dev}"        # dev  -> zero-shot eval prep (grailqa_adapter.py)
                          # train-> joint fine-tuning records (grailqa_train_adapter.py)
INPUT="${3:-}"
case "$DATASET" in
  grailqa)
    DEFAULT_DEV="$ROOT/data/grailqa/grailqa_v1.0_dev.json"
    DEFAULT_TRAIN="$ROOT/data/grailqa/grailqa_v1.0_train.json"
    ;;
  grailqa++)
    DEFAULT_DEV="$ROOT/data/grailqa++/grailqa++_dev.json"
    DEFAULT_TRAIN="$ROOT/data/grailqa++/grailqa++_train.json"
    ;;
  *) echo "Usage: $0 [grailqa|grailqa++] [dev|train] [labelled.json]" >&2; exit 2 ;;
esac
cd "$ROOT"
PYTHON="${MEMQ_PYTHON:-python}"
[[ -x .venv311/bin/python ]] && PYTHON=".venv311/bin/python"
[[ -x .venv/bin/python ]] && [[ ! -x .venv311/bin/python ]] && PYTHON=".venv/bin/python"
case "$SPLIT" in
  dev)
    INPUT="${INPUT:-$DEFAULT_DEV}"
    [[ -s "$INPUT" ]] || { echo "Missing labelled dataset: $INPUT" >&2; exit 1; }
    "$PYTHON" grailqa_adapter.py --dataset "$DATASET" --input "$INPUT"
    ;;
  train)
    INPUT="${INPUT:-$DEFAULT_TRAIN}"
    [[ -s "$INPUT" ]] || { echo "Missing labelled dataset: $INPUT" >&2; exit 1; }
    "$PYTHON" grailqa_train_adapter.py --dataset "$DATASET" --input "$INPUT"
    ;;
  *) echo "Usage: $0 [grailqa|grailqa++] [dev|train] [labelled.json]" >&2; exit 2 ;;
esac
