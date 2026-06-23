"""Offline Type-1 memory descriptions — NO API key required.

Why this exists: the canonical path is `get_key_explain.py` (DeepSeek, TYPE1_TEMPLATE),
but (a) this machine has no DEEPSEEK_API_KEY and (b) for the *no-retrain* Exp-1
checkpoint the v9 model emits the crude phrasing it was trained on
(`graph_explain.py` fallback: "the <relation> of ?entity1"). Matching the memory
descriptions to that exact phrasing maximizes retrieval for the EXISTING model.

So we deterministically build a Type-1 description per unique key, mirroring the
training fallback, and merge it into key_explain.json. For the final Exp-4 retrain,
regenerate with DeepSeek instead (proper direction-aware phrasing).

Run: python gen_type1_explain_offline.py
"""
import json
import re

NS_REL = re.compile(r'ns:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+')


def rel_words(iri):
    # ns:location.country.languages_spoken -> "languages spoken"
    return iri.split('.')[-1].replace('_', ' ')


def describe(key):
    """Mirror graph_explain.py's crude Type-1 fallback, direction-agnostic
    (the v9 model cannot express direction either)."""
    rels = NS_REL.findall(key)
    if not rels:
        return None
    if "UNION" in key and len(rels) >= 2:
        names = []
        for r in rels:
            w = rel_words(r)
            if w not in names:
                names.append(w)
        phrase = " or ".join(names)
    else:
        phrase = rel_words(rels[0])
    return f"?entity2 is the {phrase} of ?entity1."


if __name__ == "__main__":
    with open("output/all_key.json", "r") as f:
        all_key = json.load(f)

    type1_keys = list(dict.fromkeys(all_key.get("1", [])))  # unique, ordered
    key_explain1 = {}
    skipped = 0
    for k in type1_keys:
        desc = describe(k)
        if desc is None:
            skipped += 1
            continue
        key_explain1[k] = desc

    with open("output/key_explain1.json", "w") as f:
        json.dump(key_explain1, f)
    print(f"Type-1 offline descriptions: {len(key_explain1)} written, {skipped} skipped")

    # merge into the full memory: existing key_explain.json holds Type-2/3 (511)
    with open("output/key_explain.json", "r") as f:
        key_explain = json.load(f)
    before = len(key_explain)
    key_explain.update(key_explain1)
    with open("output/key_explain.json", "w") as f:
        json.dump(key_explain, f)
    print(f"key_explain.json: {before} -> {len(key_explain)} keys "
          f"({len(set(key_explain1.values()))} unique Type-1 descriptions)")
