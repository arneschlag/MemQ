#!/usr/bin/env python3
"""Round-trip check: gold GrailQA plans -> reconstructed SPARQL.

Feeds the gold plan (``sparql_explain`` from graph_explain) back through the
memq_core reconstructor as if the model had produced it, then asserts the
reconstructed query carries the expected operator (COUNT / ORDER BY / FILTER /
type anchor). This validates the reconstruction path for every new operator
without needing the trained model.

Run after the joint memory + merge_explain_data.json exist:
    MEMQ_KEY_EXPLAIN=output/key_explain_v14.json \
    MEMQ_MID_NAMES=output/All_cached_mid_names.json \
    .venv311/bin/python scripts/verify_grailqa_reconstruction.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MEMQ_KEY_EXPLAIN", "output/key_explain_v14.json")
os.environ.setdefault("MEMQ_MID_NAMES", "output/All_cached_mid_names.json")
os.environ.setdefault("MEMQ_RETRIEVAL", "adaptive")

import memq_core as mc  # noqa: E402  (import after env is set: builds the index)

data = json.load(open("output/merge_explain_data.json"))
grail = [d for d in data if str(d.get("id", "")).startswith("grailqa-")]
print(f"grailqa records with plans: {len(grail)}")

# one representative per operator class
buckets = {"count": None, "argmax": None, "argmin": None, "<": None, ">": None,
           ">=": None, "<=": None, "none_literal": None, "type_anchor": None,
           "plain": None}
for d in grail:
    fn = d.get("function", "none")
    if fn in buckets and buckets[fn] is None:
        buckets[fn] = d
    if buckets["type_anchor"] is None and any(
            t[1] == "ns:type.object.type" for t in d["where"]):
        buckets["type_anchor"] = d
    if buckets["none_literal"] is None and fn == "none" and d.get("filter"):
        buckets["none_literal"] = d
    if buckets["plain"] is None and fn == "none" and not d.get("filter") \
            and "order" not in d and d.get("aggregation") != "count":
        buckets["plain"] = d

checks = {
    "count":       lambda q: "COUNT(" in q.upper(),
    "argmax":      lambda q: "ORDER BY DESC" in q.upper(),
    "argmin":      lambda q: "ORDER BY ASC" in q.upper(),
    "<":           lambda q: "FILTER" in q and "<" in q,
    ">":           lambda q: "FILTER" in q and ">" in q,
    ">=":          lambda q: "FILTER" in q and ">=" in q,
    "<=":          lambda q: "FILTER" in q and "<=" in q,
    "none_literal":lambda q: "FILTER" in q,
    "type_anchor": lambda q: "type.object.type" in q,
    "plain":       lambda q: q.count("ns:") >= 1,
}

npass = nfail = 0
for name, d in buckets.items():
    if d is None:
        print(f"  [skip] no example for {name}")
        continue
    d = dict(d)
    # build_reconstruction re-inserts a space before every ?var (it normalizes
    # llama3 model output, which emits vars without a leading space). Gold plans
    # already have that space, so strip it to faithfully simulate model output.
    d["test_plan"] = "\n".join(
        ln for ln in d["sparql_explain"].replace(" ?", "?").split("\n") if ln.strip())
    try:
        rec = mc.build_reconstruction(d)
        q = rec["reconstruct_sparql"]
        ok = checks[name](q)
    except Exception as e:  # noqa: BLE001
        ok, q = False, f"EXCEPTION: {e}"
    status = "PASS" if ok else "FAIL"
    npass += ok
    nfail += (not ok)
    print(f"  [{status}] {name}: {d['id']}")
    print(f"        Q: {d['question']}")
    print(f"        plan: {d['sparql_explain'].strip()[:200].replace(chr(10),' | ')}")
    flat = re.sub(r"[ \n]+", " ", str(q))[:260]
    print(f"        sparql: {flat}")

print(f"\n{npass} passed, {nfail} failed")
raise SystemExit(1 if nfail else 0)
