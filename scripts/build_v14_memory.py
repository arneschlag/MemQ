#!/usr/bin/env python3
"""Publish the joint WebQSP+CWQ+GrailQA memory as the canonical v14 file.

The joint memory is produced by ``get_key_explain.py`` (DeepSeek descriptions for
Type-1/2/3 statements) into ``output/key_explain.json``. For the v14 model both
*training* (graph_explain plan targets) and *inference* (memq_core retrieval)
must use the exact same memory, so we snapshot it under a stable name and leave
the v9 memory (``output/key_explain_v9.json``) untouched.

Unlike ``build_v9_memory.py`` this does NOT rewrite Type-1 descriptions to the
offline phrasing: v14 is trained from scratch on the DeepSeek descriptions, so
they are kept verbatim (the paper's GLM-4 approach — one consistent describer for
memory and training).
"""
import json
import os
import sys

SRC = os.environ.get("MEMQ_KEY_EXPLAIN_SRC", "output/key_explain.json")
DST = sys.argv[1] if len(sys.argv) > 1 else "output/key_explain_v14.json"


def main():
    with open(SRC) as f:
        mem = json.load(f)
    n_type = sum(1 for k in mem if k.count(" .\n") == 0)
    n_type2 = sum(1 for k in mem if k.count(" .\n") == 1)
    n_type3 = sum(1 for k in mem if k.count(" .\n") == 2)
    anchor = [k for k in mem if "ns:type.object.type" in k]
    with open(DST, "w") as f:
        json.dump(mem, f)
    print(f"v14 memory -> {DST}: {len(mem)} keys "
          f"(Type-1 {n_type}, Type-2 {n_type2}, Type-3 {n_type3})")
    for k in anchor:
        print(f"  type-anchor key: {k!r} -> {mem[k]!r}")


if __name__ == "__main__":
    main()
