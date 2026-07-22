#!/usr/bin/env bash
# Build the joint WebQSP+CWQ+GrailQA memory and v14 fine-tuning corpus.
#
# Offline, Freebase-free except that WebQSP/CWQ train graphs+cvt lists must
# already exist (they are committed from the v9 build). GrailQA CVT detection
# uses the deterministic structural rule (get_cvt_list.py grailqa). DeepSeek is
# needed only to describe the new GrailQA statement templates.
#
# Prereqs:
#   * scripts/download_grailqa.sh   (data/grailqa/grailqa_v1.0_train.json)
#   * DEEPSEEK_API_KEY in .env
#   * output/{webqsp,cwq}_train_cvt_list.json present
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f .env ]] && { set -a; source .env; set +a; }
PYTHON="${MEMQ_PYTHON:-python}"
[[ -x .venv311/bin/python ]] && PYTHON=".venv311/bin/python"
[[ -x .venv/bin/python ]] && [[ ! -x .venv311/bin/python ]] && PYTHON=".venv/bin/python"

TRAIN="${1:-data/grailqa/grailqa_v1.0_train.json}"
[[ -s "$TRAIN" ]] || { echo "Missing $TRAIN (run scripts/download_grailqa.sh)" >&2; exit 1; }

echo "[1/7] GrailQA gold graph_query -> native parse records"
"$PYTHON" grailqa_train_adapter.py --input "$TRAIN" --dataset grailqa

echo "[2/7] build query graphs (reuses build_graph_train)"
"$PYTHON" - <<'PY'
import build_graph_train as b
b.build_graph("grailqa")
PY

echo "[3/7] offline CVT detection (Freebase-free structural rule)"
"$PYTHON" get_cvt_list.py grailqa

echo "[4/7] merge GrailQA entity/type names into the shared name cache"
"$PYTHON" - <<'PY'
import json
cache=json.load(open("output/All_cached_mid_names.json"))
overlay=json.load(open("output/grailqa_train_entity_names.json"))
# GrailQA darf nur ERGAENZEN, niemals ueberschreiben. Sein friendly_name ist oft
# kuerzer als der etablierte Name ("nixon" statt "Richard Nixon"); ein
# cache.update(overlay) zerschiesst damit 451 WebQSP/CWQ-Namen. Die Test-Prompts
# tragen aber die alten Namen, sodass die Entity-Aufloesung ins Leere laeuft und
# jede betroffene Rekonstruktion scheitert (20% der Faelle).
added=0
for mid, name in overlay.items():
    if mid not in cache:
        cache[mid] = name
        added += 1
json.dump(cache, open("output/All_cached_mid_names.json","w"))
print(f"name cache -> {len(cache)} entries ({added} neu aus GrailQA)")
PY

echo "[5/7] split joint corpus into Type-1/2/3 statements"
MEMQ_SPLIT_DATASETS="${MEMQ_SPLIT_DATASETS:-webqsp,cwq,grailqa}" "$PYTHON" graph_split.py

echo "[6/7] describe new statements with DeepSeek + publish v14 memory"
[[ -n "${DEEPSEEK_API_KEY:-}" ]] || { echo "DEEPSEEK_API_KEY required" >&2; exit 1; }
"$PYTHON" get_key_explain.py
"$PYTHON" scripts/build_v14_memory.py output/key_explain_v14.json

echo "[7/7] plan traces + v14 fine-tune data"
MEMQ_KEY_EXPLAIN=output/key_explain_v14.json "$PYTHON" graph_explain.py
MEMQ_FINETUNE_OUT=output/memq_finetune_data_v14.json "$PYTHON" gen_memq_finetune_data.py

echo "Done. Use output/memq_finetune_data_v14.json as the LLaMA-Factory training set."
