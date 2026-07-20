#!/usr/bin/env python3
"""Create a hop-wise MemQ evaluation table similar to the paper figure."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def mean(values):
    return sum(values) / len(values) if values else None


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def load_dataset(output, dataset, tag):
    lookup_path = output / f"{dataset}_test_lookup_{tag}.json"
    score_path = output / f"{dataset}_score_{tag}.json"
    if not lookup_path.exists() or not score_path.exists():
        return None
    with lookup_path.open() as handle:
        lookup = {item["id"]: item for item in json.load(handle)}
    with score_path.open() as handle:
        scores = json.load(handle)
    groups = defaultdict(lambda: defaultdict(list))
    for score in scores:
        item = lookup.get(score["id"], {})
        hop = score.get("hop_count") or len(item.get("where") or [])
        if not hop:
            continue
        groups[hop]["f1"].append(score["f1"])
        groups[hop]["hit"].append(score["hit@1"])
        if item.get("ehr") is not None:
            groups[hop]["ehr"].append(item["ehr"])
        if item.get("gold_ged") is not None:
            groups[hop]["ged"].append(item["gold_ged"])
    return groups


def table(ax, title, datasets, hops, metric):
    ax.axis("off")
    rows = []
    for name, groups in datasets.items():
        values = [mean(groups[hop][metric]) for hop in hops]
        # The overall benchmark metric is the mean over examples, not an
        # unweighted mean over hop bins (whose group sizes vary substantially).
        overall_values = [value for hop in hops for value in groups[hop][metric]]
        label = {"webqsp": "WebQSP", "cwq": "CWQ", "grailqa_dev": "GrailQA dev",
                 "grailqa++_dev": "GrailQA++ dev"}.get(name, name)
        rows.append([label] + [fmt(value) for value in values] + [fmt(mean(overall_values))])
    widths = [0.14] + [(0.86 / (len(hops) + 1))] * (len(hops) + 1)
    rendered = ax.table(cellText=rows, colLabels=["Dataset"] + [str(hop) for hop in hops] + ["avg"],
                        colWidths=widths, cellLoc="center", loc="center")
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(9)
    rendered.scale(1, 1.55)
    for cell in rendered.get_celld().values():
        cell.set_edgecolor("#555555")
    for col in range(len(hops) + 2):
        rendered[(0, col)].set_facecolor("#d9d9d9")
        rendered[(0, col)].set_text_props(weight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", y=1.08)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output")
    parser.add_argument("--tag", default="v9_dirfb")
    parser.add_argument("--datasets", default="webqsp,cwq,grailqa_dev,grailqa++_dev")
    parser.add_argument("--figure", default="figures/memq_hop_metrics.png")
    parser.add_argument("--summary", default="results/memq_hop_metrics.json")
    args = parser.parse_args()
    output = Path(args.output)
    loaded = {name: load_dataset(output, name, args.tag) for name in args.datasets.split(",")}
    loaded = {name: groups for name, groups in loaded.items() if groups}
    if not loaded:
        raise SystemExit("No matching lookup + score files found. Run evaluation with MEMQ_GRAPH_METRICS=1 first.")
    hops = list(range(1, max(hop for groups in loaded.values() for hop in groups) + 1))
    figure = Path(args.figure); figure.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(max(11, len(hops) * 1.25), 7))
    table(axes[0, 0], "Edge Hitting Rate (EHR)", loaded, hops, "ehr")
    table(axes[0, 1], "Gold Graph Edit Distance (GoldGED)", loaded, hops, "ged")
    table(axes[1, 0], "Answer Macro-F1", loaded, hops, "f1")
    table(axes[1, 1], "Answer Hits@1", loaded, hops, "hit")
    fig.suptitle("MemQ v9 + direction fallback by total hops", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.88, bottom=0.07,
                        hspace=0.58, wspace=0.03)
    fig.savefig(figure, dpi=220, bbox_inches="tight")
    summary = {}
    for name, groups in loaded.items():
        summary[name] = {
            str(hop): {metric: mean(values) for metric, values in metrics.items()}
            for hop, metrics in groups.items()
        }
        summary[name]["overall"] = {
            metric: mean([value for hop in groups for value in groups[hop][metric]])
            for metric in ("ehr", "ged", "f1", "hit")
        }
    summary_path = Path(args.summary); summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote {figure} and {summary_path}")


if __name__ == "__main__":
    main()
