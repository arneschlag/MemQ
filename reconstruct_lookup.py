"""MemQ lookup pass — memory retrieval + SPARQL reconstruction, **NO database**.

Runs the expensive/embedding half of MemQ entirely offline: for each test
question it retrieves memories for the model's plan, assembles the reconstructed
SPARQL, and scores Structure Accuracy against the gold `ori_sparql` (relation
multiset = hop pattern). This is the primary DB-free signal for ranking Exp 0-3
and is safe to run in the background while Virtuoso is down.

Outputs output/<DS>_test_lookup_<TAG>.json with the reconstructed SPARQL (+ two
fallbacks) per question, for the DB-dependent score pass (score_answers.py).

Config (env): MEMQ_DS (webqsp|cwq), MEMQ_PLAN (override plan file),
MEMQ_RETRIEVAL, MEMQ_GAMMA1/2, MEMQ_EMBED_MODEL, MEMQ_KEY_EXPLAIN, MEMQ_TAG.
"""
import os
import json
import memq_core
from memq_core import build_reconstruction, structure_accuracy

DS = os.environ.get("MEMQ_DS", "webqsp")
PLAN = os.environ.get("MEMQ_PLAN", f"output/{DS}_test_plan_v10.json")
TAG = os.environ.get("MEMQ_TAG",
                     f"{memq_core.RETRIEVAL}_{os.path.basename(memq_core.EMBED_MODEL)}")

with open(PLAN, "r") as f:
    testdata = json.load(f)

print(f"[lookup] DS={DS} plan={PLAN} retrieval={memq_core.RETRIEVAL} "
      f"embed={memq_core.EMBED_MODEL} memory={len(memq_core.explain_list)} keys "
      f"items={len(testdata)}", flush=True)

results = []
n = struct_hits = fails = 0
n_with_gold = struct_hits_gold = 0
for idx, d in enumerate(testdata):
    item = {k: d.get(k) for k in ("id", "question", "ori_sparql", "AnsE", "BegE",
                                  "main_path", "test_plan")}
    try:
        rec = build_reconstruction(d)
        item.update(rec)
        sa = structure_accuracy(rec["reconstruct_sparql"], d.get("ori_sparql", ""))
        item["struct_acc"] = sa
        struct_hits += sa
    except Exception as e:
        item["error"] = f"{type(e).__name__}: {e}"
        item["struct_acc"] = 0
        fails += 1
    n += 1
    if d.get("ori_sparql"):
        n_with_gold += 1
        struct_hits_gold += item["struct_acc"]
    results.append(item)
    if idx % 200 == 0 and idx > 0:
        print(f"  {idx}/{len(testdata)} struct_acc={struct_hits/n:.3f} fails={fails}", flush=True)

out = f"output/{DS}_test_lookup_{TAG}.json"
with open(out, "w") as f:
    json.dump(results, f)

print("=" * 60)
print(f"[lookup] {DS} | items={n} | reconstruction fails={fails}")
print(f"[lookup] Structure-Acc (all): {struct_hits}/{n} = {struct_hits/n:.4f}")
if n_with_gold:
    print(f"[lookup] Structure-Acc (gold present): {struct_hits_gold}/{n_with_gold} = {struct_hits_gold/n_with_gold:.4f}")
print(f"[lookup] wrote {out}")
