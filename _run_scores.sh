#!/usr/bin/env bash
# DB score pass — real Hits@1 / Macro-F1 against Virtuoso (localhost:3001).
# Gold answers are cached per dataset (output/<ds>_gold_answers.json), so only
# the first tag per dataset pays the gold-execution cost.
set -e
PY=.venv311/bin/python
for ds in webqsp cwq; do
  for tag in exp0_baseline exp1_type1 exp2_adaptive_90_80; do
    echo "########## SCORE $ds $tag ##########"
    MEMQ_DS=$ds MEMQ_TAG=$tag $PY score_answers.py
  done
done
echo "ALL_SCORES_DONE"
