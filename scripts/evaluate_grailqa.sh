#!/usr/bin/env bash
# Score already-generated plans for a GrailQA-style dev set.
#
#   scripts/evaluate_grailqa.sh [grailqa|grailqa++]
#   MEMQ_MODEL_VERSION=v14 scripts/evaluate_grailqa.sh grailqa
#
# v9  - the WebQSP/CWQ-only model, evaluated zero-shot on GrailQA (the seminar
#       report's transfer experiment).
# v14 - the joint model, trained on GrailQA as well; uses the joint memory so
#       that training and inference memory stay identical.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-grailqa}"
case "$DATASET" in grailqa|grailqa++) ;; *) echo "Usage: $0 [grailqa|grailqa++]" >&2; exit 2 ;; esac
VERSION="${MEMQ_MODEL_VERSION:-v9}"
case "$VERSION" in
  v9)  KEY_EXPLAIN="output/key_explain_v9.json"
       # build_v9_memory.py takes --output, build_v14_memory.py a positional arg.
       MEMORY_CMD=(scripts/build_v9_memory.py --output "$KEY_EXPLAIN")
       PLAN_DEFAULT="output/${DATASET}_dev_plan_v10.json"; TAG="v9_dirfb" ;;
  v14) KEY_EXPLAIN="output/key_explain_v14.json"
       MEMORY_CMD=(scripts/build_v14_memory.py "$KEY_EXPLAIN")
       PLAN_DEFAULT="output/${DATASET}_dev_plan_v14.json"; TAG="v14_dirfb" ;;
  *)   echo "Unknown MEMQ_MODEL_VERSION='$VERSION' (expected v9 or v14)" >&2; exit 2 ;;
esac
cd "$ROOT"
[[ -f .env ]] && source .env
[[ -n "${MEMQ_SPARQL_ENDPOINT:-}" ]] || { echo "Configure MEMQ_SPARQL_ENDPOINT first." >&2; exit 1; }
PYTHON="${MEMQ_PYTHON:-python}"
[[ -x .venv311/bin/python ]] && PYTHON=".venv311/bin/python"
[[ -x .venv/bin/python ]] && [[ ! -x .venv311/bin/python ]] && PYTHON=".venv/bin/python"
PLAN="${MEMQ_PLAN:-$PLAN_DEFAULT}"
[[ -s "$PLAN" ]] || { echo "Missing $PLAN. Generate plans with run_inference.py on a CUDA/ROCm machine first." >&2; exit 1; }
ENTITY_NAMES="output/${DATASET}_entity_names.json"
[[ -s "$ENTITY_NAMES" ]] || { echo "Missing $ENTITY_NAMES. Run scripts/prepare_grailqa.sh first." >&2; exit 1; }

"$PYTHON" "${MEMORY_CMD[@]}"
MERGED_NAMES="output/${DATASET}_cached_mid_names.json"
"$PYTHON" scripts/merge_entity_names.py output/All_cached_mid_names.json "$ENTITY_NAMES" "$MERGED_NAMES"
MEMQ_DS="${DATASET}_dev" MEMQ_PLAN="$PLAN" MEMQ_RETRIEVAL=adaptive \
  MEMQ_KEY_EXPLAIN="$KEY_EXPLAIN" MEMQ_MID_NAMES="$MERGED_NAMES" MEMQ_TAG="$TAG" MEMQ_GRAPH_METRICS=1 \
  "$PYTHON" reconstruct_lookup.py
MEMQ_DS="${DATASET}_dev" MEMQ_TAG="$TAG" MEMQ_DIRFB=1 "$PYTHON" score_answers.py
