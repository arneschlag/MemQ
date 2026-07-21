# MemQ reproduction

Research reproduction of [MemQ: Memory-augmented Query Reconstruction for
LLM-based Knowledge Graph Reasoning](https://aclanthology.org/2025.findings-acl.1234/).
It reconstructs executable Freebase SPARQL from a question, topic entity, and a
model-generated natural-language plan.

**Built with Meta Llama 3.** The public v9 fine-tune, its provenance, and the
applicable base-model terms are documented in [MODEL_CARD.md](MODEL_CARD.md).

## What is public

- Source code and experiment documentation in this repository.
- `Llama-3-MemQ-v9`, the merged 8B v9 model (14.97 GiB), through public B2
  URLs only; the download script validates SHA-256 checksums.
- The WebQSP/CWQ files used in this reproduction and the small generated
  artifacts needed to rerun the database-free reconstruction pass, also through
  public B2 URLs. The source datasets remain subject to their original terms.

Freebase/Virtuoso data, local databases, certificates, virtual environments,
API keys, and generated training data are intentionally **not** in this Git
repository or the release data bundle.

## Fastest reproduction: database-free lookup

This path reproduces reconstructed-query/structure metrics from the supplied
v9 plans. It needs no GPU and no Freebase service. It is the recommended first
check that an installation is correct.

```bash
git clone https://github.com/arneschlag/MemQ.git
cd MemQ

# Guided selection of CPU, NVIDIA/CUDA or AMD/ROCm; creates .venv, installs
# dependencies and downloads ~25 MiB of prompts, plans, and memory artifacts.
scripts/setup.sh

# First run downloads sentence-transformers/all-MiniLM-L6-v2 from Hugging Face.
scripts/reproduce_lookup.sh webqsp
scripts/reproduce_lookup.sh cwq
```

### Setup choices: CPU, CUDA, or ROCm

Running `scripts/setup.sh` without arguments guides the user through platform,
optional model/data downloads, and the Freebase SPARQL endpoint. It saves that
endpoint locally to ignored `.env`, then `scripts/reproduce_answers.sh` loads it
automatically. For automation, the script also accepts an explicit platform:

```bash
scripts/setup.sh cpu                 # CPU-only lookup/evaluation; smallest install
scripts/setup.sh cuda                # NVIDIA CUDA; default PyTorch cu126 wheels
scripts/setup.sh rocm                # AMD ROCm; default PyTorch rocm6.3 wheels
scripts/setup.sh auto --weights      # also download the 14.97 GiB v9 model
scripts/setup.sh cpu --raw-data      # additionally download the 65 MiB source datasets
```

For an unusual driver/runtime, choose the platform explicitly and override its
wheel index, for example
`PYTORCH_ROCM_INDEX=https://download.pytorch.org/whl/rocm6.2.4 scripts/setup.sh rocm`.
PyTorch's supported wheel combinations change over time; consult the official
[installation selector](https://pytorch.org/get-started/locally/) when the
default does not match a machine.

The resulting `output/*_test_lookup_public_v9.json` files contain reconstructed
SPARQL and database-free metrics. The original reported answer-level scores
require a local Freebase/Virtuoso service, described below.

### Reproduce the historical v9 + dirfb run

After setup and download of the public artifacts, this command rebuilds the
v9-compatible Type-1 memory deterministically, reconstructs the supplied v9
plans, and enables the historical direction fallback during answer scoring:

```bash
scripts/reproduce_v9_dirfb.sh all
```

It needs a reachable Freebase/Virtuoso endpoint, configured by `scripts/setup.sh`.
The run writes `output/{webqsp,cwq}_metrics_v9_dirfb.json`. Macro-F1 is stable
across compatible Freebase deployments; Hits@1 can vary slightly because many
Freebase `SELECT DISTINCT` results have no specified order and the metric uses
the first returned answer.

### Zero-shot GrailQA and GrailQA++ evaluation

This branch evaluates the existing v9 checkpoint only; it never fine-tunes on
either Grail dataset. GrailQA's labelled development split is the public
answer-level benchmark (the official test labels are hidden):

```bash
scripts/download_grailqa.sh
scripts/prepare_grailqa.sh grailqa

# On the CUDA/ROCm machine that has the downloaded v9 model:
MEMQ_INFERENCE_DATASETS=grailqa_dev DATA_DIR="$PWD/output" \
  MODEL_DIR="$PWD/models/Llama-3-MemQ-v9" python run_inference.py

scripts/evaluate_grailqa.sh grailqa
```

The preparation step writes a separate MID-to-friendly-name overlay from the
benchmark's supplied entity labels.  The scorer merges that overlay with the
public v9 cache; no GrailQA relation, answer, or gold plan is provided to v9.

GrailQA++ uses the same adapter, but its official repository currently ships no
dataset files. Once an authorized labelled JSON release is available, place it
at `data/grailqa++/grailqa++_dev.json`, then run the same three commands with
`grailqa++` as the argument. The adapter reports coverage separately: functions
(`count`, extrema, comparisons) and literal constraints are unsupported by the
fixed v9 plan language and are excluded rather than mislabeled as failures.

For a paper-style hop breakdown after runs with `MEMQ_GRAPH_METRICS=1`, use:

```bash
python scripts/plot_benchmark_metrics.py
```

It creates `figures/memq_hop_metrics.png`, with EHR, GoldGED, Macro-F1, and
Hits@1 tables across every evaluated dataset.

To additionally calculate the experimental EHR/GoldGED diagnostics (slower),
set `MEMQ_GRAPH_METRICS=1` before running `reconstruct_lookup.py`.

## Model inference

The supplied merged v9 model can generate plans on a CUDA/ROCm-capable machine.
Run `scripts/setup.sh cuda --weights` or `scripts/setup.sh rocm --weights` for
a clean setup, then download the model if it was not selected during setup:

```bash
scripts/download_weights.sh                 # resumable, SHA-256 verified
scripts/download_reproduction_data.sh       # provides test prompts/plans/memory

MODEL_DIR="$PWD/models/Llama-3-MemQ-v9" DATA_DIR="$PWD/output" \
  python run_inference.py
```

`run_inference.py` writes `*_test_plan_v10.json`, the filenames consumed by the
reconstruction scripts. This naming is deliberate: an older script wrote
`*_test_plan.json`, which did not match the evaluator default.

## Answer-level evaluation (requires Freebase)

Run an HTTP SPARQL endpoint containing the same Freebase/Virtuoso snapshot used
for the experiment. The endpoint defaults to `http://localhost:3001/sparql` and
is configurable without source edits:

```bash
export MEMQ_SPARQL_ENDPOINT='http://localhost:3001/sparql'
MEMQ_DS=webqsp MEMQ_TAG=public_v9 python score_answers.py
MEMQ_DS=cwq    MEMQ_TAG=public_v9 python score_answers.py
```

After guided setup, the shorter form is:

```bash
scripts/reproduce_answers.sh webqsp
scripts/reproduce_answers.sh cwq
```

The large Freebase dump/database is not redistributed here. Follow the
[DKI Freebase setup](https://github.com/dki-lab/Freebase-Setup), point
`MEMQ_SPARQL_ENDPOINT` to it, and expect answer-level scores to vary if the
snapshot, endpoint behavior, or gold-answer availability differs. Cached gold
answers are written locally to `output/`.

When running MemQ itself in Docker while Virtuoso is bound only to the Docker
host's loopback interface, start the MemQ container with `--network host` and
use `http://localhost:7001/sparql`. A normal bridge-network container cannot
reach a port published solely on `127.0.0.1`.

## Full data and training pipeline

To download the original WebQSP/CWQ input files that were used here:

```bash
scripts/download_reproduction_data.sh --raw
```

The preprocessing order is:

```text
get_my_traindata.py / get_my_testdata.py
  -> gen_parse_data.py
  -> build_graph_train.py / build_graph_test.py
  -> get_cvt_list.py          (requires Freebase)
  -> graph_split.py
  -> get_key_explain.py       (requires DEEPSEEK_API_KEY)
  -> graph_explain.py
  -> gen_memq_finetune_data.py
```

`get_key_explain.py` reads `DEEPSEEK_API_KEY` from the environment and refuses
to run when it is unset. Never place that key in a source file, `.env` file that
is committed, shell history, or a GitHub secret-scan exception. Fine-tuning is
performed with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); the
v9 configuration was LoRA rank 16, five epochs, learning rate `5e-5`, and
cutoff length 2048.

The joint v14 model (WebQSP + CWQ + GrailQA) keeps those hyperparameters and
adds sequence packing (`packing: true`, `neat_packing: true`,
`per_device_train_batch_size: 2`, `gradient_accumulation_steps: 1`), which
yields 17,390 optimizer steps — close to v9's 18,145, so the learning rate
carries over unchanged.

On RDNA4 (gfx1201) packing is not merely an optimisation: without it, batches
are padded to their own longest sequence, and the resulting varying GEMM shapes
intermittently hit a Tensile kernel that performs an illegal memory access,
killing the run. The reported error (`rocBLAS error: Could not initialize
Tensile host: No devices found`) is misleading — by then the HIP context is
already gone, so consult `dmesg` for the actual fault. Switching BLAS backend
(`TORCH_BLAS_PREFER_HIPBLASLT`) does not help, since both libraries dispatch to
the same Tensile kernels. Packing makes every batch exactly
`batch × cutoff_len`, which keeps the shapes constant and the run stable; it
also removes a large amount of padding waste and roughly halves the runtime.

## Results

The complete result matrix and methodology notes are in
[RESULTS.md](RESULTS.md). The key result is that fixing the missing Type-1
memory keys was decisive; v9 was the best overall model in this reproduction.
Results should be treated as reproduction measurements, not as a claim of exact
paper parity: the Freebase snapshot and evaluation coverage differ.

## Publication and security notes

- No live API keys, S3 credentials, private database files, or certificates are
  tracked. `.gitignore` explicitly excludes them. Before publishing a fork,
  scan the entire Git history as well as the working tree.
- The original upstream repository did not include an explicit software
  license. This repository therefore does **not** grant a new permissive license
  for upstream code; retain upstream attribution and obtain permission before
  relicensing it.
- The model weights are a derivative of Meta Llama 3, not a separately
  permissively licensed artifact. See [MODEL_CARD.md](MODEL_CARD.md).

## Citation

```bibtex
@inproceedings{xu-etal-2025-memory,
  title={Memory-augmented Query Reconstruction for {LLM}-based Knowledge Graph Reasoning},
  author={Xu, Mufan and Liang, Gewen and Chen, Kehai and Wang, Wei and Zhou, Xun and Yang, Muyun and Zhao, Tiejun and Zhang, Min},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
  year={2025}
}
```
