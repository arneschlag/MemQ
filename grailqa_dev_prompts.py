#!/usr/bin/env python3
"""Build GrailQA dev evaluation prompts WITH operator coverage.

The v9-era ``grailqa_adapter.py`` skips count / extrema / comparison / literal
questions, so the sample produced from it contains only ``function: none`` and
cannot show whether the operator extension works. This script reuses the
converter written for training (``grailqa_train_adapter.convert``), which
handles the full operator set, and turns its records into the prompt format the
inference script consumes.

No fine-tuning involved: the operators are already in the trained model: what
was missing is a test set that exercises them.

    .venv311/bin/python grailqa_dev_prompts.py \
        --input data/grailqa/grailqa_v1.0_dev.json \
        --out output/grailqa_dev_prompt.json
"""
import argparse
import collections
import json

from grailqa_train_adapter import convert

# Same wording as gen_memq_finetune_data.py, otherwise the model sees a prompt
# format it was never trained on.
INPUT_TEMPLATE = """You are given a problem to solve step by step. Each step should begin with either "Find", "Make sure" or "Rank". Finally, you need to output which one is the final answer. The steps should logically follow from one another, where each step builds on the outcome of the previous steps.
Each step should be simple, clear, and directly related to achieving the overall goal. Some topic entities you can use to start the plan are provided below.

Question:
{question}

Topic Entity:
{topic_entity}

Plan:
"""


def gold_answers(row):
    out = []
    for a in row.get("answer", []):
        v = a.get("answer_argument")
        if v is None:
            continue
        out.append("ns:" + v if a.get("answer_type") == "entity" else str(v))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/grailqa/grailqa_v1.0_dev.json")
    ap.add_argument("--out", default="output/grailqa_dev_prompt.json")
    ap.add_argument("--names-out", default="output/grailqa_dev_entity_names.json")
    ap.add_argument("--dataset", default="grailqa")
    args = ap.parse_args()

    rows = json.load(open(args.input))
    prepared, names, skipped = [], {}, collections.Counter()

    for row in rows:
        res = convert(row, args.dataset)
        if res[0] is None:
            skipped[res[1]] += 1
            continue
        record, _, row_names = res
        names.update(row_names)
        topic = row_names.get(record["BegE"], record["BegE"])
        record["prompt"] = INPUT_TEMPLATE.format(question=record["question"],
                                                 topic_entity="*" + topic + "*")
        record["gold_answers"] = gold_answers(row)
        prepared.append(record)

    json.dump(prepared, open(args.out, "w"))
    json.dump(names, open(args.names_out, "w"))

    ops = collections.Counter(r.get("function", "none") for r in prepared)
    lv = collections.Counter(r.get("level", "unknown") for r in prepared)
    print(f"{len(prepared)}/{len(rows)} converted -> {args.out}")
    if skipped:
        print("skipped:", dict(skipped))
    print("level   :", dict(lv))
    print("function:", dict(ops))
    print(f"with an operator: {sum(v for k, v in ops.items() if k != 'none')}")
    print(f"entity names -> {args.names_out} ({len(names)})")


if __name__ == "__main__":
    main()
