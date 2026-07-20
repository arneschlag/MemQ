#!/usr/bin/env python3
"""Prepare GrailQA-style JSON for zero-shot MemQ evaluation.

No example from GrailQA is used for fine-tuning.  Gold graph annotations are
used only to identify the supplied topic entity and to evaluate reconstructed
queries, matching the known-entity assumption used for WebQSP/CWQ.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx


PROMPT = """You are given a problem to solve step by step. Each step should begin with either \"Find\", \"Make sure\" or \"Rank\". Finally, you need to output which one is the final answer. The steps should logically follow from one another, where each step builds on the outcome of the previous steps.
Each step should be simple, clear, and directly related to achieving the overall goal. Some topic entities you can use to start the plan are provided below.

Question:
{question}

Topic Entity:
*{topic_entity}*

Plan:
"""


def _term(node, question_nid):
    if node["nid"] == question_nid:
        return "?x"
    if node["node_type"] == "entity":
        return "ns:" + node["id"]
    return f"?v{node['nid']}"


def _answers(row):
    values = []
    for answer in row.get("answer", []):
        value = answer.get("answer_argument")
        if not value:
            continue
        values.append("ns:" + value if answer.get("answer_type") == "Entity" else str(value))
    return values


def convert(row, dataset):
    """Return a MemQ item or ``(None, reason)`` for unsupported graph forms."""
    if row.get("function", "none") != "none":
        return None, "function"
    graph = row.get("graph_query")
    if not graph:
        return None, "missing_graph"
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    question_nodes = [node for node in nodes if node.get("question_node") == 1]
    if len(question_nodes) != 1:
        return None, "question_node"
    if any(node.get("node_type") == "literal" for node in nodes):
        return None, "literal"
    question_nid = question_nodes[0]["nid"]
    node_by_id = {node["nid"]: node for node in nodes}
    entities = [node for node in nodes if node.get("node_type") == "entity"]
    if not entities:
        return None, "no_topic_entity"

    ug = nx.Graph()
    ug.add_nodes_from(node_by_id)
    ug.add_edges_from((edge["start"], edge["end"]) for edge in edges)
    reachable = []
    for entity in entities:
        try:
            path = nx.shortest_path(ug, entity["nid"], question_nid)
        except nx.NetworkXNoPath:
            continue
        reachable.append((len(path), entity, path))
    if not reachable:
        return None, "topic_not_connected"
    # Prefer the longest grounded-to-answer path, as the original graph builder
    # does for its main path. Tie-break by graph node id for determinism.
    _, topic, path = max(reachable, key=lambda item: (item[0], -item[1]["nid"]))
    where = []
    for edge in edges:
        try:
            relation = "ns:" + edge["relation"]
            where.append([_term(node_by_id[edge["start"]], question_nid), relation,
                          _term(node_by_id[edge["end"]], question_nid)])
        except KeyError:
            return None, "malformed_edge"

    qid = str(row.get("qid", row.get("id", "unknown")))
    topic_name = topic.get("friendly_name") or topic["id"]
    return {
        "id": f"{dataset}-{qid}",
        "dataset": dataset,
        "question": row["question"],
        "ori_sparql": row.get("sparql_query", ""),
        # GrailQA gold SPARQL projects ?value.  Plans retain MemQ's ?x
        # convention; score_answers.py keeps the two answer variables separate.
        "gold_AnsE": "?value",
        "AnsE": "?x",
        "BegE": "ns:" + topic["id"],
        "main_path": [_term(node_by_id[nid], question_nid) for nid in path],
        "where": where,
        "hop_count": len(edges),
        "level": row.get("level", "unknown"),
        "function": row.get("function", "none"),
        "gold_answers": _answers(row),
        "prompt": PROMPT.format(question=row["question"], topic_entity=topic_name),
    }, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="GrailQA/GrailQA++ labelled JSON array")
    parser.add_argument("--dataset", required=True, choices=("grailqa", "grailqa++"))
    parser.add_argument("--output", help="Default: output/<dataset>_dev_prompt.json")
    parser.add_argument("--report", help="Default: output/<dataset>_dev_preparation.json")
    args = parser.parse_args()
    dataset_file = Path(args.input)
    with dataset_file.open() as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array")

    prepared, skipped = [], Counter()
    for row in rows:
        item, reason = convert(row, args.dataset)
        if item is None:
            skipped[reason] += 1
        else:
            prepared.append(item)
    output = Path(args.output or f"output/{args.dataset}_dev_prompt.json")
    report = Path(args.report or f"output/{args.dataset}_dev_preparation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(prepared, handle)
    with report.open("w") as handle:
        json.dump({"dataset": args.dataset, "input": str(dataset_file),
                   "total": len(rows), "prepared": len(prepared),
                   "skipped": dict(sorted(skipped.items()))}, handle, indent=2)
    print(f"Prepared {len(prepared)}/{len(rows)} {args.dataset} examples -> {output}")
    if skipped:
        print("Skipped:", dict(sorted(skipped.items())))


if __name__ == "__main__":
    main()
