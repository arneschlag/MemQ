#!/usr/bin/env python3
"""Merge the public v9 name cache with entity names supplied by a benchmark."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="Existing MID -> name JSON cache")
    parser.add_argument("overlay", help="Benchmark-supplied MID -> name JSON cache")
    parser.add_argument("output", help="Merged cache path")
    args = parser.parse_args()
    with open(args.base) as handle:
        merged = json.load(handle)
    with open(args.overlay) as handle:
        overlay = json.load(handle)
    merged.update(overlay)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        json.dump(merged, handle)
    print(f"Wrote {target}: {len(merged)} entity names")


if __name__ == "__main__":
    main()
