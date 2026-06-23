#!/usr/bin/env bash
# Exp 2 (adaptive recall) + Exp 3 (mpnet embedding) DB-free lookup, on the
# Type-1-fixed memory (key_explain.json). Structure-Acc is the offline ranking
# signal. NOTE: gamma sweep here is over TEST structure-acc (exploratory) — a
# clean dev-tuned value needs model-generated dev plans (bigstore).
set -e
export HF_HUB_OFFLINE=1
PY=.venv311/bin/python

for ds in webqsp cwq; do
  echo "########## $ds Exp2 adaptive g1=0.90 g2=0.80 ##########"
  MEMQ_DS=$ds MEMQ_RETRIEVAL=adaptive MEMQ_GAMMA1=0.90 MEMQ_GAMMA2=0.80 \
    MEMQ_TAG=exp2_adaptive_90_80 $PY reconstruct_lookup.py

  echo "########## $ds Exp2 adaptive g1=0.85 g2=0.70 ##########"
  MEMQ_DS=$ds MEMQ_RETRIEVAL=adaptive MEMQ_GAMMA1=0.85 MEMQ_GAMMA2=0.70 \
    MEMQ_TAG=exp2_adaptive_85_70 $PY reconstruct_lookup.py

  echo "########## $ds Exp3 mpnet (legacy retrieval) ##########"
  MEMQ_DS=$ds MEMQ_EMBED_MODEL=model/all-mpnet-base-v2 \
    MEMQ_TAG=exp3_mpnet_legacy $PY reconstruct_lookup.py

  echo "########## $ds Exp3 mpnet + adaptive g1=0.90 g2=0.80 ##########"
  MEMQ_DS=$ds MEMQ_EMBED_MODEL=model/all-mpnet-base-v2 MEMQ_RETRIEVAL=adaptive \
    MEMQ_GAMMA1=0.90 MEMQ_GAMMA2=0.80 MEMQ_TAG=exp3_mpnet_adaptive_90_80 \
    $PY reconstruct_lookup.py
done
echo "ALL_EXP23_DONE"
