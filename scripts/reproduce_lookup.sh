#!/usr/bin/env bash
# Reproduce the database-free reconstruction metrics from public artifacts.
set -euo pipefail

DATASET="${1:-webqsp}"
if [[ "$DATASET" != "webqsp" && "$DATASET" != "cwq" ]]; then
  echo "Usage: $0 [webqsp|cwq]" >&2
  exit 2
fi

if [[ ! -f output/key_explain.json || ! -f output/All_cached_mid_names.json || ! -f "output/${DATASET}_test_plan_v10.json" ]]; then
  echo "Missing public reproduction artifacts. Run scripts/download_reproduction_data.sh first." >&2
  exit 1
fi

MEMQ_DS="$DATASET" MEMQ_RETRIEVAL=adaptive MEMQ_TAG="public_v9" \
  python reconstruct_lookup.py
