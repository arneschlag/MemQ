"""Inference runner for MemQ — runs inside llamafactory container on bigstore.
Reads *_test_prompt.json, generates test plans with the fine-tuned model,
writes *_test_plan.json. Logs progress to /root/data/inference.log
"""
import json
import time
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_DIR = os.environ.get("MODEL_DIR", "/root/models/memq-llama3-8b-lora-v9-merged")
DATA_DIR = os.environ.get("DATA_DIR", "/root/data")
PLAN_SUFFIX = os.environ.get("MEMQ_PLAN_SUFFIX", "v10")
DATASETS = [name.strip() for name in os.environ.get(
    "MEMQ_INFERENCE_DATASETS", "webqsp_test,cwq_test"
).split(",") if name.strip()]

def load_model(path):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    print(f"[load] {time.time()-t0:.0f}s  device={next(model.parameters()).device}", flush=True)
    return model, tok

def generate_plans(data, model, tokenizer, dataset_label=""):
    ok = retry = 0
    for i, d in enumerate(tqdm(data, desc=dataset_label)):
        prompt = d.get("prompt", "")
        if not prompt:
            continue
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attn = ids.ne(tokenizer.pad_token_id).long()
        # Retry loop from gen_testplan.py
        plan = ""
        error_cnt = 0
        while error_cnt < 3 and (not plan.startswith("Step1")):
            out = model.generate(
                ids, attention_mask=attn,
                max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            plan = tokenizer.decode(out[0][len(ids[0]):], skip_special_tokens=True).strip()
            if not plan.startswith("Step1"):
                error_cnt += 1
        d["test_plan"] = plan
        if plan.startswith("Step1"):
            ok += 1
        elif error_cnt >= 3:
            retry += 1
        # Print every 200th so we can see progress in logs
        if i % 200 == 0 and i > 0:
            print(f"  [{dataset_label}] {i}/{len(data)} ok={ok} retry_fail={retry}", flush=True)
    print(f"  [{dataset_label}] done: {len(data)} plans, ok={ok}, retry_fail={retry}", flush=True)
    return data

if __name__ == "__main__":
    t_start = time.time()

    model, tokenizer = load_model(MODEL_DIR)

    for name in DATASETS:
        prompt_path = os.path.join(DATA_DIR, f"{name}_prompt.json")
        plan_path = os.path.join(DATA_DIR, f"{name}_plan_{PLAN_SUFFIX}.json")
        print(f"\n=== {name} ===", flush=True)
        with open(prompt_path, "r") as f:
            data = json.load(f)
        print(f"  loaded {len(data)} prompts", flush=True)

        data = generate_plans(data, model, tokenizer, dataset_label=name)

        with open(plan_path, "w") as f:
            json.dump(data, f)
        print(f"  saved {plan_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\ntotal inference time: {elapsed/3600:.1f}h ({elapsed/60:.0f}min)",
          flush=True)
