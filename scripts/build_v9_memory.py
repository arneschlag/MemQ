#!/usr/bin/env python3
"""Build the historical v9 retrieval memory without an API key.

The v9 checkpoint was trained with the deterministic Type-1 phrasing used by
the original ``graph_explain.py`` fallback. Later DeepSeek descriptions stay
in place for Type-2/3 keys, while Type-1 descriptions are restored to their
v9-compatible form.
"""
import argparse
import json
import re
from pathlib import Path


RELATION = re.compile(r"ns:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def describe_type1(key: str) -> str:
    relations = RELATION.findall(key)
    if not relations:
        raise ValueError(f"No Freebase relation in Type-1 key: {key!r}")
    if "UNION" in key and len(relations) >= 2:
        words = []
        for relation in relations:
            name = relation.rsplit(".", 1)[-1].replace("_", " ")
            if name not in words:
                words.append(name)
        phrase = " or ".join(words)
    else:
        phrase = relations[0].rsplit(".", 1)[-1].replace("_", " ")
    return f"?entity2 is the {phrase} of ?entity1."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/key_explain.json")
    parser.add_argument("--output", default="output/key_explain_v9.json")
    args = parser.parse_args()
    with open(args.input) as handle:
        memory = json.load(handle)

    type1 = [key for key in memory if key.count(" .\n") == 0]
    memory.update({key: describe_type1(key) for key in type1})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(memory, handle)
    print(f"Wrote {output}: {len(memory)} keys, {len(type1)} Type-1 entries restored")


if __name__ == "__main__":
    main()
