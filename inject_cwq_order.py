#!/usr/bin/env python3
"""Add the missing ORDER BY / LIMIT (Rank) information to CWQ training records.

gen_parse_data.py drops the sort clause while parsing, so the ``order`` field is
empty for every CWQ record even though 1916 of them carry ORDER BY ... LIMIT in
the gold query. The model therefore never sees a Rank step and cannot produce
superlatives (272 CWQ test questions, F1 0.42 vs ~0.80). GrailQA is unaffected:
its converter sets ``order`` directly.

The sort variable and its binding triple survive parsing (only the ORDER line is
dropped), so the spec can be recovered from ori_sparql and injected into the
split records with no Freebase round trip. graph_explain already turns ``order``
into a Rank step (it does so for GrailQA), and memq_core already reconstructs it.

    MEMQ_SPLIT_IN=output/merge_split_data.json \
    MEMQ_SPLIT_OUT=output/merge_split_data_v15.json python inject_cwq_order.py
"""
import json
import os
import re

IN = os.environ.get("MEMQ_SPLIT_IN", "output/merge_split_data.json")
OUT = os.environ.get("MEMQ_SPLIT_OUT", "output/merge_split_data_v15.json")

CWQ_ID = re.compile(r"WebQ.*_[0-9a-f]{16,}$")
ORDER = re.compile(
    r"ORDER\s+BY\s+(.*?)\s+LIMIT\s+(\d+)(?:\s+OFFSET\s+(\d+))?", re.I | re.S)


def extract_order(sparql, where):
    m = ORDER.search(sparql or "")
    if not m:
        return None
    expr = m.group(1).strip()
    direction = "DESC" if re.match(r"DESC", expr, re.I) else "ASC"
    expr = re.sub(r"^(DESC|ASC)\s*", "", expr, flags=re.I).strip()
    v = re.findall(r"\?[A-Za-z0-9_]+", expr)
    if not v:
        return None
    # The sort variable must be bound somewhere in the graph, else the Rank step
    # would reference a variable no Find step produced.
    if not any(v[0] in str(t) for t in (where or [])):
        return None
    # graph_explain reads "datetime"/"float"/"integer" from `var` to type the
    # Rank step, but for an uncast key it wants the bare variable ("?num", not
    # "(?num)") -- otherwise its var[0]=="?" check fails and it raises.
    has_cast = any(c in expr.lower() for c in ("datetime", "float", "integer"))
    var = expr if has_cast else v[0]
    return {"order": direction, "var": var,
            "start": int(m.group(3)) if m.group(3) else 0, "len": int(m.group(2))}


def main():
    data = json.load(open(IN))
    injected = skipped = 0
    for rec in data:
        if not CWQ_ID.match(str(rec.get("id", ""))):
            continue
        if not re.search(r"ORDER\s+BY", rec.get("ori_sparql", "") or "", re.I):
            continue
        o = extract_order(rec["ori_sparql"], rec.get("where"))
        if o:
            rec["order"] = o
            injected += 1
        else:
            skipped += 1
    json.dump(data, open(OUT, "w"))
    print(f"injected order into {injected} CWQ records "
          f"({skipped} skipped) -> {OUT}")


if __name__ == "__main__":
    main()
