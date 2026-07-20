#!/usr/bin/env bash
# Reconstruct and score one dataset using the endpoint saved by scripts/setup.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-webqsp}"
if [[ "$DATASET" != "webqsp" && "$DATASET" != "cwq" ]]; then
  echo "Usage: $0 [webqsp|cwq]" >&2
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

MEMQ_DS="$DATASET" MEMQ_RETRIEVAL=adaptive MEMQ_TAG="fresh_v9" \
  "$PYTHON" reconstruct_lookup.py
MEMQ_DS="$DATASET" MEMQ_TAG="fresh_v9" "$PYTHON" score_answers.py
