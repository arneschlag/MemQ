#!/usr/bin/env python3
"""Convert GrailQA gold graph queries into MemQ *training* parse-records.

Unlike ``grailqa_adapter.py`` (which prepares a zero-shot *evaluation* subset and
drops every example using a function or a literal), this adapter targets the
joint fine-tuning corpus and keeps the full GrailQA operator set:

* path queries (function ``none``)                       -> ordinary Find steps
* extrema  (``argmax`` / ``argmin``)                     -> d['order'] (Sort step)
* comparison (``<`` ``>`` ``<=`` ``>=``)                 -> casted FILTER (Make sure)
* literal equality (typed literal object, function none) -> equality FILTER
* count                                                  -> d['aggregation']='count'
* type-anchored questions (no topic entity)              -> synthetic
      ``?x ns:type.object.type ns:<class>`` triple, with the class friendly name
      registered so the reconstructor resolves it like a topic entity.

The output matches the ``{dataset}_train_parse.json`` schema consumed by
``build_graph_train.build_graph`` so the rest of the offline pipeline
(``graph_split`` -> ``get_key_explain`` -> ``graph_explain`` ->
``gen_memq_finetune_data``) runs unchanged.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


# Freebase XSD datatype URIs used in GrailQA literal node ids / classes.
_XSD = "http://www.w3.org/2001/XMLSchema#"


def _split_literal(node):
    """Return (raw_value, dtype) for a literal node.

    GrailQA encodes literals as ``"<value>^^<uri>"`` in ``id`` and the coarse
    kind in ``class`` (``type.int`` / ``type.float`` / ``type.datetime`` ...).
    """
    nid = node.get("id", "")
    value, _, uri = nid.partition("^^")
    uri = uri or ""
    cls = node.get("class", "")
    if "date" in uri or cls == "type.datetime":
        dtype = "date"
    elif "float" in uri or "double" in uri or "decimal" in uri or cls == "type.float":
        dtype = "float"
    elif "int" in uri or cls in ("type.int", "type.integer"):
        dtype = "integer"
    else:
        dtype = "string"
    return value, dtype


def _comparison_filter(var, op, value, dtype):
    """Canonical casted FILTER string that explain_filter/process_filter parse."""
    if dtype == "integer":
        return f'xsd:integer({var}) {op} "{value}"^^<{_XSD}integer>'
    if dtype == "float":
        return f'xsd:float({var}) {op} "{value}"^^<{_XSD}float>'
    if dtype == "date":
        return f'xsd:datetime({var}) {op} "{value}"^^xsd:dateTime'
    # string comparison is not meaningful for <,> — fall back to str()
    return f'str({var}) {op} "{value}"'


def _equality_filter(var, value, dtype):
    """Canonical equality FILTER for a bound literal object (function none)."""
    if dtype == "date":
        return f'{var} = "{value}"^^<{_XSD}dateTime>'
    if dtype == "float":
        return f'{var} = "{value}"'
    if dtype == "integer":
        return f'{var} = "{value}"'
    return f'{var} = "{value}"'


def convert(row, dataset):
    """Return (record, reason). record is None on an unconvertible row."""
    graph = row.get("graph_query")
    if not graph:
        return None, "missing_graph"
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    q_nodes = [n for n in nodes if n.get("question_node") == 1]
    if len(q_nodes) != 1:
        return None, "question_node"
    q_nid = q_nodes[0]["nid"]
    node_by_id = {n["nid"]: n for n in nodes}

    def term(nid):
        n = node_by_id[nid]
        if nid == q_nid:
            return "?x"
        if n.get("node_type") == "entity":
            return "ns:" + n["id"]
        return f"?v{nid}"

    where = []
    for e in edges:
        try:
            where.append([term(e["start"]), "ns:" + e["relation"], term(e["end"])])
        except KeyError:
            return None, "malformed_edge"

    filters = []
    order = None
    aggregation = None
    type_names = {}

    # question-node aggregation
    if node_by_id[q_nid].get("function") == "count":
        aggregation = "count"

    # literal-node operators
    for n in nodes:
        if n.get("node_type") != "literal":
            continue
        var = term(n["nid"])
        fn = n.get("function", "none")
        value, dtype = _split_literal(n)
        if fn in ("<", ">", "<=", ">="):
            filters.append(_comparison_filter(var, fn, value, dtype))
        elif fn == "argmax":
            order = {"order": "DESC", "var": var, "start": 0, "len": 1}
        elif fn == "argmin":
            order = {"order": "ASC", "var": var, "start": 0, "len": 1}
        elif fn == "none":
            filters.append(_equality_filter(var, value, dtype))
        else:
            return None, f"unhandled_function:{fn}"

    # topic entity vs. type anchor
    entities = [n for n in nodes if n.get("node_type") == "entity"]
    entity_names = {n["nid"]: n.get("friendly_name") or n["id"] for n in entities}
    if entities:
        beg_e = "ns:" + entities[0]["id"]  # build_graph_train re-selects longest path
    else:
        # No topic entity: anchor on the answer node's Freebase type. A synthetic
        # type.object.type edge lets the anchor behave like a topic entity through
        # the whole pipeline (type.object.type is already a guard relation).
        anchor = node_by_id[q_nid]
        cls = anchor.get("class") or anchor.get("id")
        if not cls:
            return None, "no_anchor"
        type_iri = "ns:" + cls
        where.append(["?x", "ns:type.object.type", type_iri])
        type_name = anchor.get("friendly_name") or cls.split(".")[-1].replace("_", " ")
        type_names[type_iri] = type_name
        beg_e = type_iri

    qid = str(row.get("qid", row.get("id", "unknown")))
    record = {
        "id": f"{dataset}-{qid}",
        "dataset": dataset,
        "question": row["question"],
        "ori_sparql": row.get("sparql_query", ""),
        "BegE": beg_e,
        "AnsE": "?x",
        "gold_AnsE": "?value",
        "where": where,
        "filter": filters,
        "exists": [],
        "level": row.get("level", "unknown"),
        "function": row.get("function", "none"),
    }
    if order is not None:
        record["order"] = order
    if aggregation is not None:
        record["aggregation"] = aggregation
    # friendly-name overlay: entities + synthetic type anchors
    names = {"ns:" + n["id"]: (n.get("friendly_name") or n["id"]) for n in entities}
    names.update(type_names)
    return record, None, names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="GrailQA labelled train JSON array")
    ap.add_argument("--dataset", default="grailqa")
    ap.add_argument("--output", help="Default: output/<dataset>_train_parse.json")
    ap.add_argument("--names", help="Default: output/<dataset>_train_entity_names.json")
    ap.add_argument("--report", help="Default: output/<dataset>_train_preparation.json")
    args = ap.parse_args()

    rows = json.load(open(args.input))
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON array")

    prepared, skipped, names = [], Counter(), {}
    op_counts = Counter()
    for row in rows:
        result = convert(row, args.dataset)
        if result[0] is None:
            skipped[result[1]] += 1
            continue
        record, _, row_names = result
        prepared.append(record)
        names.update(row_names)
        op_counts[record.get("function", "none")] += 1
        if "order" in record:
            op_counts["_order"] += 1
        if record.get("aggregation") == "count":
            op_counts["_count"] += 1
        if any(t[1] == "ns:type.object.type" for t in record["where"]):
            op_counts["_type_anchor"] += 1

    out = Path(args.output or f"output/{args.dataset}_train_parse.json")
    names_out = Path(args.names or f"output/{args.dataset}_train_entity_names.json")
    report = Path(args.report or f"output/{args.dataset}_train_preparation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(prepared, open(out, "w"))
    json.dump(names, open(names_out, "w"))
    json.dump({"dataset": args.dataset, "input": args.input, "total": len(rows),
               "prepared": len(prepared), "skipped": dict(sorted(skipped.items())),
               "operators": dict(sorted(op_counts.items()))}, open(report, "w"), indent=2)
    print(f"Prepared {len(prepared)}/{len(rows)} -> {out}")
    print("operators:", dict(sorted(op_counts.items())))
    if skipped:
        print("skipped:", dict(sorted(skipped.items())))


if __name__ == "__main__":
    main()
