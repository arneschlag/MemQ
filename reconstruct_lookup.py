"""MemQ lookup pass — memory retrieval + SPARQL reconstruction, **NO database**.

Runs the expensive/embedding half of MemQ entirely offline: for each test
question it retrieves memories for the model's plan, assembles the reconstructed
SPARQL, and scores Structure Accuracy against the gold `ori_sparql` (relation
multiset = hop pattern). This is the primary DB-free signal for ranking Exp 0-3
and is safe to run in the background while Virtuoso is down.

Outputs output/<DS>_test_lookup_<TAG>.json with the reconstructed SPARQL (+ two
fallbacks) per question, for the DB-dependent score pass (score_answers.py).

Config (env): MEMQ_DS (webqsp|cwq), MEMQ_PLAN (override plan file),
MEMQ_RETRIEVAL, MEMQ_GAMMA1/2, MEMQ_EMBED_MODEL, MEMQ_KEY_EXPLAIN, MEMQ_TAG,
MEMQ_GRAPH_METRICS=1 (optional EHR/GoldGED; slower).
"""
import os
import json
import memq_core
from memq_core import build_reconstruction, structure_accuracy

DS = os.environ.get("MEMQ_DS", "webqsp")
PLAN = os.environ.get("MEMQ_PLAN", f"output/{DS}_test_plan_v10.json")
TAG = os.environ.get("MEMQ_TAG",
                     f"{memq_core.RETRIEVAL}_{os.path.basename(memq_core.EMBED_MODEL)}")
GRAPH_METRICS = os.environ.get("MEMQ_GRAPH_METRICS", "0") == "1"

with open(PLAN, "r") as f:
    testdata = json.load(f)

print(f"[lookup] DS={DS} plan={PLAN} retrieval={memq_core.RETRIEVAL} "
      f"embed={memq_core.EMBED_MODEL} memory={len(memq_core.explain_list)} keys "
      f"items={len(testdata)} graph_metrics={GRAPH_METRICS}", flush=True)

results = []
n = struct_hits = fails = 0
n_with_gold = struct_hits_gold = 0
sum_ehr = ehr_cnt = 0
sum_ged = ged_cnt = 0
for idx, d in enumerate(testdata):
    item = {k: d.get(k) for k in ("id", "dataset", "question", "ori_sparql", "AnsE",
                                  "gold_AnsE", "gold_answers", "BegE", "main_path",
                                  "test_plan", "where", "hop_count", "level", "function")}
    try:
        rec = build_reconstruction(d)
        item.update(rec)
        sa = structure_accuracy(rec["reconstruct_sparql"], d.get("ori_sparql", ""))
        item["struct_acc"] = sa
        struct_hits += sa
        if GRAPH_METRICS:
            ehr, gold_ged = memq_core.reasoning_metrics(
                d.get("where"), rec["reconstruct_sparql"]
            )
            item["ehr"] = ehr
            item["gold_ged"] = gold_ged
            if ehr is not None:
                sum_ehr += ehr
                ehr_cnt += 1
            if gold_ged is not None:
                sum_ged += gold_ged
                ged_cnt += 1
    except Exception as e:
        item["error"] = f"{type(e).__name__}: {e}"
        item["struct_acc"] = 0
        if GRAPH_METRICS:
            item["ehr"] = None
            item["gold_ged"] = None
        fails += 1
    n += 1
    if d.get("ori_sparql"):
        n_with_gold += 1
        struct_hits_gold += item["struct_acc"]
    results.append(item)
    if idx % 200 == 0 and idx > 0:
        extra = ""
        if GRAPH_METRICS:
            extra = (f" ehr={(sum_ehr/ehr_cnt if ehr_cnt else float('nan')):.3f}"
                     f" ged={(sum_ged/ged_cnt if ged_cnt else float('nan')):.3f}")
        print(f"  {idx}/{len(testdata)} struct_acc={struct_hits/n:.3f} fails={fails}{extra}", flush=True)

out = f"output/{DS}_test_lookup_{TAG}.json"
with open(out, "w") as f:
    json.dump(results, f)

avg_ehr = sum_ehr / ehr_cnt if ehr_cnt else float('nan')
avg_ged = sum_ged / ged_cnt if ged_cnt else float('nan')

print("=" * 60)
print(f"[lookup] {DS} | items={n} | reconstruction fails={fails}")
print(f"[lookup] Structure-Acc (all): {struct_hits}/{n} = {struct_hits/n:.4f}")
if n_with_gold:
    print(f"[lookup] Structure-Acc (gold present): {struct_hits_gold}/{n_with_gold} = {struct_hits_gold/n_with_gold:.4f}")
if GRAPH_METRICS:
    print(f"[lookup] EHR (n={ehr_cnt}): {avg_ehr:.4f}")
    print(f"[lookup] GoldGED (n={ged_cnt}): {avg_ged:.4f}")
print(f"[lookup] wrote {out}")
