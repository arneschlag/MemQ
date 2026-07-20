"""MemQ score pass — execute reconstructed SPARQL, compute Hits@1 + Macro-F1.

This is the ONLY DB-dependent step: it reads the lookup output (which already
holds the reconstructed SPARQL) and runs gold + predicted queries against the
Freebase Virtuoso service. Torch-free on purpose — it only needs requests.

Paper-faithful evaluation:
  * Macro-F1 = mean of per-sample F1 (not F1 of the averaged P/R).
  * Failures / empty predictions score 0 and STAY in the denominator.
  * Gold answers are cached to output/<DS>_gold_answers.json so re-runs of later
    experiments don't re-execute the gold side.

Denominator: by default the full set of questions that HAVE a gold answer set
(`--all` to score over every question, counting unanswerable gold as 0).

Config (env): MEMQ_DS, MEMQ_TAG (which lookup file), MEMQ_SCORE_ALL=1 for full-set,
MEMQ_DIRFB=1 to enable v9's direction fallback. Per-example scores are written
to ``output/<DS>_score_<TAG>.json`` for hop-level reports.
Run only when Virtuoso is up:  python score_answers.py
"""
import os
import sys
import json
from sparql_util import get_result

DS = os.environ.get("MEMQ_DS", "webqsp")
TAG = os.environ.get("MEMQ_TAG", "legacy_all-MiniLM-L6-v2")
SCORE_ALL = os.environ.get("MEMQ_SCORE_ALL", "0") == "1" or "--all" in sys.argv
# Historical v9 compatibility: only try reconstruct_sparql3 when explicitly
# requested, so ordinary modern runs retain their original evaluation path.
DIRFB = os.environ.get("MEMQ_DIRFB", "0") == "1"

LOOKUP = f"output/{DS}_test_lookup_{TAG}.json"
GOLD_CACHE = f"output/{DS}_gold_answers.json"


def eval_result(true_list, pred_list):
    true_set, pred_set = set(true_list), set(pred_list)
    inter = true_set & pred_set
    precision = len(inter) / len(pred_set) if pred_set else 0.0
    recall = len(inter) / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    hit = 1 if (pred_list and pred_list[0] in true_set) else 0
    return precision, recall, f1, hit


def safe_exec(sparql, ansvar):
    if not sparql:
        return []
    try:
        return get_result(sparql, ansvar)
    except Exception:
        return []


def main():
    with open(LOOKUP, "r") as f:
        data = json.load(f)

    gold_cache = {}
    if os.path.exists(GOLD_CACHE):
        with open(GOLD_CACHE, "r") as f:
            gold_cache = json.load(f)

    n_total = len(data)
    scored = 0
    no_gold = 0
    sum_f1 = sum_hit = sum_p = sum_r = 0.0
    details = []

    for i, d in enumerate(data):
        qid = d.get("id") or str(i)
        pred_ans = d.get("ansE") or d.get("AnsE") or "?x"
        gold_ans = d.get("gold_AnsE") or d.get("AnsE") or "?x"
        # --- gold (cached) ---
        if d.get("gold_answers") is not None:
            gold = d["gold_answers"]
        elif qid in gold_cache:
            gold = gold_cache[qid]
        else:
            gold = safe_exec(d.get("ori_sparql"), gold_ans)
            gold_cache[qid] = gold
            if (i + 1) % 100 == 0:
                with open(GOLD_CACHE, "w") as f:
                    json.dump(gold_cache, f)

        if not gold:
            no_gold += 1
            if not SCORE_ALL:
                continue
            # score over full set: unanswerable gold contributes 0
            scored += 1
            details.append({"id": qid, "hop_count": d.get("hop_count", len(d.get("where") or [])),
                            "level": d.get("level"), "f1": 0.0, "hit@1": 0,
                            "precision": 0.0, "recall": 0.0, "no_gold": True})
            continue

        # --- predicted: primary -> fallback1 -> fallback2 -> optional v9 dirfb ---
        pred = safe_exec(d.get("reconstruct_sparql"), pred_ans)
        if not pred:
            pred = safe_exec(d.get("reconstruct_sparql1"), pred_ans)
        if not pred:
            pred = safe_exec(d.get("reconstruct_sparql2"), pred_ans)
        if not pred and DIRFB:
            pred = safe_exec(d.get("reconstruct_sparql3"), pred_ans)

        p, r, f1, hit = eval_result(gold, pred)
        sum_p += p; sum_r += r; sum_f1 += f1; sum_hit += hit
        scored += 1
        details.append({"id": qid, "hop_count": d.get("hop_count", len(d.get("where") or [])),
                        "level": d.get("level"), "f1": f1, "hit@1": hit,
                        "precision": p, "recall": r, "no_gold": False})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_total} macroF1={sum_f1/scored:.3f} hit@1={sum_hit/scored:.3f}", flush=True)

    with open(GOLD_CACHE, "w") as f:
        json.dump(gold_cache, f)

    denom = scored if scored else 1
    macro_f1 = sum_f1 / denom
    hit_at_1 = sum_hit / denom
    metrics = {
        "dataset": DS, "tag": TAG, "score_all": SCORE_ALL,
        "n_total": n_total, "scored": scored, "no_gold": no_gold,
        "hit@1": round(hit_at_1, 4), "macro_f1": round(macro_f1, 4),
        "avg_precision": round(sum_p / denom, 4), "avg_recall": round(sum_r / denom, 4),
    }
    print("=" * 60)
    print(json.dumps(metrics, indent=2))
    with open(f"output/{DS}_metrics_{TAG}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"output/{DS}_score_{TAG}.json", "w") as f:
        json.dump(details, f)


if __name__ == "__main__":
    main()
