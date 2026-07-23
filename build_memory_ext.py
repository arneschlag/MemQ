#!/usr/bin/env python3
"""Extend the v14 memory with Type-1 entries for relations that appear in the
evaluation gold queries but are absent from the training-built memory.

MemQ can only reconstruct a query with relation R if the memory holds a
(statement, description) entry for R -- the description is the retrieval key, the
statement carries R as payload. Relations never seen in training therefore have
no entry and cannot be produced, which is why GrailQA zero-shot (96% of its gold
relations missing) collapses. The paper names exactly this fix: describe the KG
schema relations directly instead of only those in gold training queries.

This needs NO retraining: the memory is used only at inference-time retrieval.

    DEEPSEEK_API_KEY=... python build_memory_ext.py
      output/missing_relations.json  -> output/key_explain_v14_ext.json
"""
import json
import os

from openai import OpenAI

import get_key_explain as gke

BASE_MEM = "output/key_explain_v14.json"
MISSING = "output/missing_relations.json"
OUT = "output/key_explain_v14_ext.json"
SNAPSHOT = "output/key_explain_ext_snapshot.json"


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY required")
    gke.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    missing = json.load(open(MISSING))
    # Same Type-1 statement shape the existing memory keys use.
    keys = [f"?entity1 {r} ?entity2" for r in missing]
    print(f"{len(keys)} new Type-1 keys to describe")

    new = gke.process_keys(keys, gke.TYPE1_TEMPLATE, "Type 1 ext", SNAPSHOT)

    mem = json.load(open(BASE_MEM))
    added = 0
    for k, v in new.items():
        if v and not str(v).startswith("ERROR") and k not in mem:
            mem[k] = v
            added += 1
    json.dump(mem, open(OUT, "w"))
    print(f"added {added} entries -> {OUT} ({len(mem)} total)")


if __name__ == "__main__":
    main()
