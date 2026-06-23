#!/usr/bin/env bash
# Exp 4 data regeneration — run AFTER a DeepSeek key is available at ~/.deepseek_key.
# Produces proper, direction-aware Type-1 memory descriptions and rebuilds the
# fine-tuning data, so the v11 retrain learns phrasing consistent with the memory.
set -e
PY=.venv311/bin/python

if [ -z "$DEEPSEEK_API_KEY" ] && [ -f ~/.deepseek_key ]; then
  export DEEPSEEK_API_KEY="$(tr -d '[:space:]' < ~/.deepseek_key)"
fi
if [ -z "$DEEPSEEK_API_KEY" ]; then
  echo "ERROR: no DEEPSEEK_API_KEY (env) or ~/.deepseek_key file"; exit 1
fi

# The offline Type-1 descriptions sit in key_explain1.json and would be treated as
# a cache by get_key_explain.py (skipping the API). Move them aside so DeepSeek
# regenerates all Type-1 keys properly.
[ -f output/key_explain1.json ] && mv output/key_explain1.json output/key_explain1.offline.json
# Restore the pre-Type-1 memory (Type-2/3 only) so the merge is clean.
cp output/key_explain.json.bak output/key_explain.json

echo "=== get_key_explain.py (DeepSeek): Type-1 descriptions ==="
$PY get_key_explain.py            # -> output/key_explain.json  (~1039 keys)

echo "=== graph_explain.py: regenerate reasoning steps (Type-1 now DeepSeek) ==="
$PY graph_explain.py              # -> output/merge_explain_data.json

echo "=== gen_memq_finetune_data.py: rebuild fine-tuning data ==="
$PY gen_memq_finetune_data.py     # -> output/memq_finetune_data.json

echo "REGEN_DONE — upload output/memq_finetune_data.json to bigstore and retrain v11"
