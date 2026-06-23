# MemQ — Experiment Results (DB-free checkpoint)

These are the cheap, GPU-free, **database-free** rungs of the plan
(`~/.claude/plans/memq-pipeline-paper-ancient-lecun.md`). The metric is
**Structure Accuracy** — the reconstructed query's relation multiset (hop
pattern) exactly equals the gold `ori_sparql`'s. It is a strict 0/1 offline
proxy (the paper's answer-level Hits@1/F1 needs the Freebase service and gives
partial credit, so the absolute numbers here understate the eventual answer
metrics — the **deltas** are what rank the fixes).

All runs use the existing **v9** model plans (`*_test_plan_v10.json`); no retrain.

## Answer-level metrics (real, via Virtuoso) — the headline

Macro-F1 = mean of per-sample F1 (paper-faithful); failures score 0. Scored over
questions with a non-empty gold answer from our local Freebase (WebQSP no_gold=11,
CWQ no_gold=258 — local-KB completeness; the paper uses official gold).

| Dataset | Config | Hits@1 | Macro-F1 |
|---|---|---|---|
| WebQSP | Exp 0 baseline (the bug) | 0.3143 | 0.3065 |
| WebQSP | **Exp 1 Type-1 fix** | **0.8389** | **0.8214** |
| WebQSP | Exp 2 + adaptive recall | 0.8395 | 0.8214 |
| WebQSP | *Paper (Llama2-7b / Llama3-8b)* | *0.841 / 0.858* | *0.858 / 0.872* |
| CWQ | Exp 0 baseline (the bug) | 0.1424 | 0.1572 |
| CWQ | **Exp 1 Type-1 fix** | **0.6734** | **0.6837** |
| CWQ | Exp 2 + adaptive recall | 0.6700 | 0.6812 |
| CWQ | *Paper (Llama2-7b)* | *0.803* | *0.830* |

**Headlines:**
- The **Type-1 fix alone** lifts WebQSP Hits@1 0.314 → **0.839** and CWQ 0.142 → **0.673** — *no retrain*. The single missing `all_key["1"].append(key)` line was crippling the whole system.
- **WebQSP Hits@1 is already at paper parity** (0.840 vs 0.841). WebQSP F1 (0.821) and all of CWQ still trail by ~0.04 / ~0.13 → the remaining gap is the v9 train/inference phrasing mismatch, which the **v11 retrain** (data already regenerated) targets.
- **Adaptive recall is neutral on answer-level at untuned γ** (±0.003 vs legacy) despite a small structure-acc gain. Kept as the default (paper-faithful); realizing a gain needs dev-tuned γ₁/γ₂ (i.e. model-generated dev plans on bigstore).

## Structure Accuracy (DB-free proxy used to rank Exp 0-3 offline)

| # | Config | Memory | Retrieval | Embedding | WebQSP | CWQ |
|---|---|---|---|---|---|---|
| Exp 0 | baseline (the bug) | 511 keys | legacy reranker | MiniLM | 0.2065 | 0.0804 |
| Exp 1 | **Type-1 fix** | 1039 keys | legacy reranker | MiniLM | **0.4404** | **0.2246** |
| Exp 2 | Type-1 + adaptive recall (γ₁.90/γ₂.80) | 1039 | adaptive (paper) | MiniLM | 0.4435 | 0.2334 |
| Exp 2 | Type-1 + adaptive recall (γ₁.85/γ₂.70) | 1039 | adaptive (paper) | MiniLM | **0.4441** | **0.2334** |
| Exp 3 | Type-1 + mpnet (legacy retrieval) | 1039 | legacy reranker | mpnet | 0.3170 | 0.0748 |
| Exp 3 | Type-1 + mpnet (adaptive) | 1039 | adaptive | mpnet | 0.4441 | 0.2328 |

## Conclusions

1. **The Type-1 memory fix is the dominant lever: +113% WebQSP (0.21→0.44), +179% CWQ (0.08→0.22).** One missing line in `graph_split.py` was halving the memory and corrupting reconstruction for the 83% of questions with a single-triple step.
2. **Adaptive recall (the paper's Eq. 4) beats the custom common-words reranker** by a small, consistent margin (WebQSP +0.8%, CWQ +3.9%). γ is insensitive (0.90/0.80 ≈ 0.85/0.70). Adopt it — it's the paper's method. *(γ shown here was explored on test; a clean dev-tuned value needs model-generated dev plans, i.e. bigstore.)*
3. **mpnet (768-dim) gives no benefit over MiniLM (384-dim):** equal with adaptive retrieval, worse with the legacy thresholds (which don't transfer to mpnet's similarity scale). **Keep all-MiniLM-L6-v2** (smaller, faster).

**Best DB-free config:** Type-1 fix + adaptive recall + MiniLM → WebQSP 0.444, CWQ 0.233.

## Why still far from the paper's 0.84 / 0.80

These are answer-level numbers in the paper; ours is a stricter structural proxy, AND the v9 model itself was trained *with* the Type-1 bug (crude phrasing) on buggy data. The remaining gap is the **train/inference mismatch** — closed by Exp 4 (regenerate training data with proper Type-1 descriptions + retrain). That is the gated, expensive rung.

## How to reproduce (all DB-free, backgroundable)

```bash
# memory rebuild (already done): fix graph_split.py, then
.venv311/bin/python graph_split.py                 # all_key.json: 528 T1 keys
.venv311/bin/python gen_type1_explain_offline.py   # offline Type-1 descriptions (no API)

# lookup pass (no database) — Structure Accuracy
HF_HUB_OFFLINE=1 MEMQ_DS=webqsp MEMQ_RETRIEVAL=adaptive \
  .venv311/bin/python reconstruct_lookup.py
```

## What still needs external resources (gated)

| Step | Needs | Status |
|---|---|---|
| Answer-level Hits@1 / Macro-F1 (`score_answers.py`) | Freebase Virtuoso at `localhost:3001` | code ready, DB down |
| Proper Type-1 descriptions (`get_key_explain.py`) | `DEEPSEEK_API_KEY` | no key on machine |
| Exp 4 v11 retrain | bigstore GPU + regenerated data | gated on go-ahead |

Backups of the pre-fix memory: `output/key_explain.json.bak`, `output/all_key.json.bak`.
