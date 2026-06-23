set -e
export HF_HUB_OFFLINE=1
for ds in webqsp cwq; do
  echo "########## $ds Exp0 baseline (511-key memory) ##########"
  MEMQ_DS=$ds MEMQ_KEY_EXPLAIN=output/key_explain.json.bak MEMQ_TAG=exp0_baseline \
    .venv311/bin/python reconstruct_lookup.py
  echo "########## $ds Exp1 Type-1 fix (1039-key memory) ##########"
  MEMQ_DS=$ds MEMQ_KEY_EXPLAIN=output/key_explain.json MEMQ_TAG=exp1_type1 \
    .venv311/bin/python reconstruct_lookup.py
done
echo "ALL_LOOKUP_DONE"
