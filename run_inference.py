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
BATCH_SIZE = int(os.environ.get("MEMQ_BATCH_SIZE", "1"))

def load_model(path):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    print(f"[load] {time.time()-t0:.0f}s  device={next(model.parameters()).device}", flush=True)
    return model, tok

def _generate_batch(prompts, model, tokenizer):
    """Greedy batched decoding; output is identical in intent to the old loop."""
    rendered = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    ) for prompt in prompts]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
    try:
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    finally:
        tokenizer.padding_side = previous_padding_side
    input_length = inputs.input_ids.shape[1]
    return [tokenizer.decode(row[input_length:], skip_special_tokens=True).strip() for row in out]


def _generate_one(prompt, model, tokenizer):
    # Retain the historical retry behavior only for malformed batch outputs.
    plan = ""
    for _ in range(3):
        plan = _generate_batch([prompt], model, tokenizer)[0]
        if plan.startswith("Step1"):
            break
    return plan


def generate_plans(data, model, tokenizer, dataset_label="", checkpoint_path=None):
    ok = retry = 0
    for start in tqdm(range(0, len(data), BATCH_SIZE), desc=dataset_label):
        batch = data[start:start + BATCH_SIZE]
        prompts = [item.get("prompt", "") for item in batch]
        plans = _generate_batch(prompts, model, tokenizer)
        for item, prompt, plan in zip(batch, prompts, plans):
            if prompt and not plan.startswith("Step1"):
                plan = _generate_one(prompt, model, tokenizer)
            item["test_plan"] = plan
            if plan.startswith("Step1"):
                ok += 1
            else:
                retry += 1
        completed = min(start + len(batch), len(data))
        if checkpoint_path and completed % 100 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(data, f)
        if completed % 200 == 0:
            print(f"  [{dataset_label}] {completed}/{len(data)} ok={ok} retry_fail={retry}", flush=True)
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

        data = generate_plans(data, model, tokenizer, dataset_label=name,
                              checkpoint_path=plan_path + ".partial")

        with open(plan_path, "w") as f:
            json.dump(data, f)
        print(f"  saved {plan_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\ntotal inference time: {elapsed/3600:.1f}h ({elapsed/60:.0f}min)",
          flush=True)
