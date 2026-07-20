#!/usr/bin/env bash
# Reproduce the historical v9 + direction-fallback evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-all}"
if [[ "$DATASET" != "webqsp" && "$DATASET" != "cwq" && "$DATASET" != "all" ]]; then
  echo "Usage: $0 [webqsp|cwq|all]" >&2
  exit 2
fi
cd "$ROOT"
if [[ -f .env ]]; then source .env; fi
if [[ -z "${MEMQ_SPARQL_ENDPOINT:-}" ]]; then
  echo "No Freebase endpoint configured. Run scripts/setup.sh and enter its URL, or export MEMQ_SPARQL_ENDPOINT." >&2
  exit 1
fi
PYTHON="${MEMQ_PYTHON:-python}"
if [[ -x .venv/bin/python ]]; then PYTHON=".venv/bin/python"; fi

"$PYTHON" scripts/build_v9_memory.py
datasets=("$DATASET")
if [[ "$DATASET" == "all" ]]; then datasets=(webqsp cwq); fi

for ds in "${datasets[@]}"; do
  echo "=== Historical v9 + dirfb: $ds ==="
  MEMQ_DS="$ds" MEMQ_RETRIEVAL=adaptive MEMQ_KEY_EXPLAIN=output/key_explain_v9.json \
    MEMQ_TAG=v9_dirfb "$PYTHON" reconstruct_lookup.py
  MEMQ_DS="$ds" MEMQ_TAG=v9_dirfb MEMQ_DIRFB=1 "$PYTHON" score_answers.py
done
