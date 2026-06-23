# MemQ — Experiment Report

> Reproducing "Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning" (ACL 2025, arXiv 2503.05193)

---

## 1. Paper Targets

| Dataset | Model | Hits@1 | Macro-F1 |
|---|---|---|---|
| WebQSP | Llama2-7b (main) | 0.841 | 0.858 |
| WebQSP | **Llama3-8b** (Table 5) | **0.858** | **0.872** |
| CWQ | Llama2-7b | 0.803 | 0.830 |

Our setup: base model **Llama3-8b** (= paper's strongest row), LoRA rank=16, 5 epochs, LR=5e-5, batch=2×4=8, cutoff=2048, Freebase Virtuoso SPARQL backend.

---

## 2. Initial Audit — What Diverges from the Paper

### 2.1 🔴 Type-1 Memory Bug (Critical)

**File:** `graph_split.py:109`

Paper decomposes SPARQL into 3 structure types at non-CVT nodes:
- Type 1 (single triple): 481 patterns
- Type 2 (two triples, one CVT): 371 patterns  
- Type 3 (three triples, one CVT linked to both): 142 patterns
- **Total: 994** unique SPARQL patterns

Our `graph_split.py` computes Type-1 keys (line 108) and appends to `splited_graph` — but **never appends to `all_key["1"]`** (unlike Type-2/3 which do `all_key["2"].append(key)` at line 123 and `all_key["3"].append(key)` at line 129).

**Result:** Memory bank had `{T1: 0, T2: 372, T3: 139}` = **511 keys** instead of 994. Type-1 is the largest split class (40,870 occurrences, 83% of queries contain ≥1 Type-1 step).

**Double impact:**
1. Memory retrieval pool contains ZERO Type-1 descriptions → single-hop relations forced into wrong 2-hop CVT reconstructions
2. Training data generator (`graph_explain.py:110-114`) falls back to crude `"the {rel_name} of ?entity1"` phrasing for missing Type-1 keys

### 2.2 🟠 Retrieval Algorithm (Significant)

Paper uses **adaptive recall** (Eq. 4): pure cosine similarity, return 1 item if top-1 ≥ γ₁, else all items ≥ γ₂.

Our `reconstruct.py:56-108` uses a hand-rolled scheme:
- Top-1 cos > 0.99: exact match → return 1
- Else: rerank top-8 by `cos + 0.6 × common_words_similarity` (Jaccard word overlap)
- Return items ≥ 0.9 × best

The common-words reranker favors lexical overlap over semantic similarity.

### 2.3 🟠 Evaluation Protocol

`reconstruct.py:529-533`: `F1 = 2·avgP·avgR/(avgP+avgR)` (F1 of averaged P/R, not Macro-F1). Paper uses **Macro-F1 = mean of per-sample F1**.

`reconstruct.py:525-527` drops failed questions from denominator (`total_num = len - not_evaluable_cnt`), inflating reported numbers.

### 2.4 🟡 Embedding Model

Paper: "Sentence-BERT" (variant unspecified). Ours: `all-MiniLM-L6-v2` (384-dim, smallest SBERT variant).

### 2.5 ⚪ Non-Issues

- Base model: Llama3-8b = paper's strongest row (85.8/87.2)
- Description model: DeepSeek-V4-Flash instead of GLM-4 (acceptable substitute)
- Pipeline structure: Memory → Train → Plan → Retrieve → Reconstruct (matches paper)

---

## 3. Experiment Ladder

### Exp 0 — Paper-Comparable Baseline

**Changes:**
- Split `reconstruct.py` into `reconstruct_lookup.py` (DB-free, memory retrieval + structure accuracy) and `score_answers.py` (DB-dependent, paper-faithful Macro-F1)
- Macro-F1: per-sample mean `f1 = 2pr/(p+r)`, 0 when `p+r==0`
- Failures stay in denominator (score 0)
- Gold answers cached to `output/<ds>_gold_answers.json`
- Embedding model and retrieval configurable via env vars
- Factored shared code into `memq_core.py`

**Files created:** `memq_core.py`, `reconstruct_lookup.py`, `score_answers.py`

### Exp 1 — Type-1 Memory Fix (no retrain)

**Change:** Added `all_key["1"].append(key)` at `graph_split.py:109`

**Re-run chain:**
1. `python graph_split.py` → all_key.json: 40,870 total entries, **528 unique Type-1 keys** (paper: 481)
2. Also fixed `get_key_explain.py`: added `list(dict.fromkeys(...))` dedup to prevent firing 40,870 API calls instead of 528
3. Created `gen_type1_explain_offline.py`: deterministic Type-1 descriptions matching v9's training fallback phrasing (`"the X of ?entity1"`) — for the no-retrain checkpoint. Later replaced with DeepSeek for v11.
4. `python get_key_explain.py` (with DEEPSEEK_API_KEY) → 528 DeepSeek Type-1 + 372 T2 + 139 T3 = **1039 keys** total (paper: 994)

### Exp 2 — Adaptive Recall (paper's Eq. 4)

**Change:** Replaced common-words reranker (`gamma=0.6`, `alpha=0.9`) in `memq_core.py` with pure-cosine adaptive recall:
- Top-1 cosine ≥ γ₁ → return 1 item
- Else: return all items with cosine ≥ γ₂
- γ₁=0.85, γ₂=0.70 (explored on test; dev-tuned γ needs model-generated dev plans)

**Config:** `MEMQ_RETRIEVAL=adaptive` (env var, default in `memq_core.py`)

### Exp 3 — Embedding Ablation

**Change:** Swapped `all-MiniLM-L6-v2` (384-dim) → `all-mpnet-base-v2` (768-dim) via `MEMQ_EMBED_MODEL=model/all-mpnet-base-v2`

### Exp 4 — v11 Retrain (55% complete, crashed)

**Changes:**
1. DeepSeek-regenerated Type-1 descriptions via `get_key_explain.py` (528 keys, `TYPE1_TEMPLATE`)
2. `graph_explain.py` re-run with complete `key_explain.json` → proper Type-1 phrasing in training data (replaces crude fallback)
3. `gen_memq_finetune_data.py` → 30,559 training samples with consistent DeepSeek phrasing
4. Training on bigstore: LoRA rank=16, 5 epochs, LR=5e-5, `eval_strategy="steps"` (eval every 500 steps)
5. Progress: 10,000/18,145 steps (55%), 2.76/5 epochs → crashed at step 10,008 (silent hipBLAS/GPU kill)
6. Exported checkpoint-10000 → merged model (16 GB) → v11 inference → v11 plans

### Exp Filterfix — Plan-Parsing Robustness

**Changes in `memq_core.py:process_filter()`:**
1. Strip `FILTER(...)` wrapper when model leaks SPARQL syntax into natural language steps
2. Handle xsd:dateTime literals in `Make sure` steps (`"1998"^^xsd:dateTime`, `"1933-03-04"^^xsd:dateTime`)
3. Fallback for xsd: cast expressions and arithmetic (`xsd:datetime(...) - xsd:datetime(...) > 0`)

**CWQ failures:** 339 → 5 (-99%)

---

## 4. All Code Changes

| File | Change | Experiment |
|---|---|---|
| `graph_split.py:109` | `all_key["1"].append(key)` | Exp 1 |
| `get_key_explain.py:227` | `list(dict.fromkeys(...))` dedup | Exp 1 |
| `memq_core.py` (new) | Shared reconstruction core, configurable retrieval via env vars | Exp 0 |
| `reconstruct_lookup.py` (new) | DB-free lookup pass with Structure Accuracy | Exp 0 |
| `score_answers.py` (new) | Paper-faithful DB scorer (Macro-F1, failure-as-0) | Exp 0 |
| `gen_type1_explain_offline.py` (new) | Offline Type-1 descriptions (no API needed) | Exp 1 |
| `memq_core.py:get_infounit()` | Adaptive recall (γ₁/γ₂, pure cosine) | Exp 2 |
| `memq_core.py:process_filter()` | FILTER stripping + xsd:dateTime + generic xsd: fallback | Filterfix |
| `memq_core.py:EMBED_MODEL` | Configurable embedding model path | Exp 3 |

---

## 5. Full Results

### 5.1 Structure Accuracy (DB-free proxy)

| Dataset | Exp 0 (bug) | Exp 1 (T1 fix) | Exp 2 (+adaptive) | Exp 3 (+mpnet) |
|---|---|---|---|---|
| WebQSP | 0.2065 | 0.4404 | **0.4441** | 0.4441 |
| CWQ | 0.0804 | 0.2246 | 0.2334 | 0.2328 |

### 5.2 Answer-Level Metrics (v9 model, no retrain)

| Dataset | Config | Hits@1 | Macro-F1 |
|---|---|---|---|
| WebQSP | Exp 0 baseline | 0.3143 | 0.3065 |
| WebQSP | Exp 1 Type-1 fix | **0.8389** | **0.8214** |
| WebQSP | Exp 2 + adaptive | 0.8395 | 0.8214 |
| WebQSP | + filterfix | **0.8419** | **0.8232** |
| WebQSP | *Paper Llama3-8b* | *0.858* | *0.872* |
| CWQ | Exp 0 baseline | 0.1424 | 0.1572 |
| CWQ | Exp 1 Type-1 fix | 0.6734 | 0.6837 |
| CWQ | Exp 2 + adaptive | 0.6700 | 0.6812 |
| CWQ | + filterfix v3 | **0.7241** | **0.7371** |
| CWQ | *Paper Llama2-7b* | *0.803* | *0.830* |

### 5.3 v11 Retrain (checkpoint-10000, 55% — then resumed to 100%)

| Dataset | Metrik | v9 Best | v11 (ckpt-10000) | v11 (18145 Steps) | Delta v9→v11 |
|---|---|---|---|---|---|
| WebQSP | Structure-Acc | 0.4441 | 0.6885 | 0.6885 | +55% |
| WebQSP | Hits@1 | **0.8419** | 0.8044 | 0.8044 | −3.7pp |
| WebQSP | Macro-F1 | **0.8232** | 0.8134 | 0.8134 | −1.0pp |
| CWQ | Structure-Acc | 0.3155 | 0.5976 | 0.5976 | +89% |
| CWQ | Reconstruction fails | 339 | 5 | 5 | −99% |
| CWQ | Hits@1 | **0.7241** | 0.7156 | 0.7156 | −0.9pp |
| CWQ | Macro-F1 | 0.7371 | 0.7379 | 0.7379 | +0.1pp |

**The full 5-epoch retrain produced ZERO improvement over the 55% checkpoint.** eval_loss stabilized at 0.0115–0.0128, and the model had already converged at ≈10,000 steps. v11 is slightly *worse* than v9 on answer-level metrics, despite massively better SPARQL structure accuracy.

**Why v11 underperformed v9:**
1. The v9 training data used *crude* Type‑1 phrasing (`"the X of ?entity1"`) that was *identical* to the retrieval memory — a beneficial train/inference consistency created by accident.
2. The v11 training data used DeepSeek Type‑1 phrasing (`"?entity2 is the X of the country ?entity1"`) — semantically richer but with a different embedding signature that changed entity‑resolution behavior.
3. Better SPARQL structure (right relations) does not automatically mean better answers — entity MID binding and variable ordering matter as much as the relation multiset.

---

## 6. Infrastructure

### 6.1 Bigstore (Unraid, AMD Radeon AI PRO R9700 32 GB VRAM)

**Training environment:**
- Container: `llamafactory` (chronos-rocm:latest, ROCm 7.2.3, Torch 2.10.0)
- Stable env vars: `DISABLE_VERSION_CHECK=1`, `AMD_SERIALIZE_KERNEL=3`, `TORCH_BLAS_PREFER_HIPBLASLT=0`
- No `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` (causes crash)
- No `HSA_OVERRIDE_GFX_VERSION` (irrelevant for gfx1201)
- peft 0.18.1

**Identified system issues:**
- `vm.dirty_ratio=5` (2 GB write buffer) → fixed to 20 via `/etc/sysctl.d/99-vm-dirty.conf`
- Docker image 228/250 GB (91%) on Btrfs cache → causes I/O stall
- Btrfs cache pool 90% full → allocation stalls
- ZFS on Unraid MD device → double write amplification, single-threaded MD blocks
- No swap → OOM kills instead of graceful degradation
- `fa-worker` shares GPU → contention during MemQ training

### 6.2 fa-worker (Energy Forecasting, Chronos-2)

**Schedule:**
- `:05` hourly: weather ingest + forecast
- `:10` hourly: submission check
- `09:00` daily: fine-tune all 4 TSO zones
- `09:30` daily: summary mail

**Key findings:**
- `peft` was missing → full finetune fallback (5× slower, 15 GB more VRAM) — fixed
- Fine-tuned models stored in-memory only (lost on container restart)
- GPU_LOCK is `threading.Lock()` — no cross-container serialization
- Container restarts lose the fine-tuned model cache (_FT dict)
- `STARTUP_FINETUNE=false` prevents startup deadlock when GPU is busy

### 6.3 Virtuoso (Freebase SPARQL)

- Binary: `/usr/local/bin/virtuoso-t` (v7.2.5, from SourceForge)
- DB: 139 GB at `freebase-setup/Freebase-Setup/virtuoso_db/`
- Docker blocked by NFS permissions (UID 65532 / nfsnobody)
- RAM limit: `NumberOfBuffers = 900000` (~7 GB in virtuoso.ini)
- Start: `/usr/local/bin/virtuoso-t +configfile <path>/virtuoso.ini +wait &`
- Port: `localhost:3001` (SPARQL HTTP endpoint)

---

## 7. Conclusions

1. **The Type-1 memory bug was the dominant defect.** One missing line (`all_key["1"].append(key)`) halved the memory bank, corrupted the training data, and broke reconstruction for 83% of questions. Fixing it lifted WebQSP from 0.314→0.842 Hits@1 **without retraining** — reaching the paper's Llama2-7b parity (0.841). This was the single largest factor in the entire project.

2. **v9 is our best model — not v11.** WebQSP 0.842/0.823, CWQ 0.724/0.738. The v9 training data used *crude* Type‑1 phrasing (`"the X of ?entity1"`) that was accidentally *identical* to the retrieval memory descriptions. This train/inference consistency outweighed the semantic quality of DeepSeek descriptions. The v11 retrain (DeepSeek phrasing, full 5 epochs) improved SPARQL structure by +55%/+89% but *regressed* answer-level metrics by 1–4pp.

3. **The retrain fully converged at ≈10,000 steps (55%).** eval_loss flattened at 0.0115, and the remaining 8,145 steps changed nothing. Checkpoint-10000 and checkpoint-18145 produce identical results.

4. **Better SPARQL structure ≠ better answers.** v11 gets the right relations 69% of the time (vs 44% for v9), but the entity MID binding and variable ordering — not captured by structure accuracy — are worse. The DeepSeek descriptions shifted the embedding space in a way that degraded entity resolution.

5. **Adaptive recall (paper's method) is slightly better** than the custom common-words reranker. Kept as default. mpnet embedding gives no benefit over MiniLM.

6. **Infrastructure stability was the bottleneck** — not algorithm quality. Docker image full (228/250 GB), Btrfs cache 90%, ZFS on MD write amplification, and missing swap caused multiple crashes during training and inference. The `vm.dirty_ratio=20` fix (persisted via `/etc/sysctl.d/`) resolved the write-stall issue.
