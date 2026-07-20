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
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Downloads ~14 MiB of plans and memory artifacts, no credentials required.
scripts/download_reproduction_data.sh

# First run downloads sentence-transformers/all-MiniLM-L6-v2 from Hugging Face.
scripts/reproduce_lookup.sh webqsp
scripts/reproduce_lookup.sh cwq
```

The resulting `output/*_test_lookup_public_v9.json` files contain reconstructed
SPARQL and database-free metrics. The original reported answer-level scores
require a local Freebase/Virtuoso service, described below.

To additionally calculate the experimental EHR/GoldGED diagnostics (slower),
set `MEMQ_GRAPH_METRICS=1` before running `reconstruct_lookup.py`.

## Model inference

The supplied merged v9 model can generate plans on a CUDA/ROCm-capable machine.
Install the PyTorch build appropriate for the system before installing the rest
of the requirements, then download the model:

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

The large Freebase dump/database is not redistributed here. Follow the
[DKI Freebase setup](https://github.com/dki-lab/Freebase-Setup), point
`MEMQ_SPARQL_ENDPOINT` to it, and expect answer-level scores to vary if the
snapshot, endpoint behavior, or gold-answer availability differs. Cached gold
answers are written locally to `output/`.

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
