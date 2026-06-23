from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
import torch

def get_model_output(prompt, model, tokenizer):
    # add_generation_prompt=True: ohne das generiert das Modell selbst einen
    # "assistant"-Header, der dem Plan vorangestellt wird (bricht den Step1-Check)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    # lora 需要截取
    generated_text = tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True).strip()
    return generated_text

# Gemergtes Modell (LoRA in Basismodell gemergt). Auf bigstore im llamafactory-Container.
model_dir = os.environ.get("MODEL_DIR", "/root/models/memq-llama3-8b-lora-v9-merged")
# dtype=bfloat16: sonst lädt das 8B-Modell als float32 (32GB) und wird auf CPU
# ausgelagert -> ~40s statt ~9s pro Generierung.
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


with open("output/webqsp_test_prompt.json", "r") as f:
    webqspdata = json.load(f)

print("generate test plan for webqsp test")
for d in tqdm(webqspdata):
    prompt = d['prompt']
    plan = get_model_output(prompt, tokenizer=tokenizer, model=model)
    error_cnt = 0
    while error_cnt<3 and plan[:5] != "Step1":
        plan = get_model_output(prompt, tokenizer=tokenizer, model=model)
        error_cnt +=1
    if plan[:5] != "Step1":
        print(f"unable to generate vaild plan of {d['id']}")
        print(plan)
    d['test_plan'] = plan

with open("output/webqsp_test_plan.json", "w") as f:
    json.dump(webqspdata, f)


    

with open("output/cwq_test_prompt.json", "r") as f:
    cwqdata = json.load(f)
print("generate test plan for cwq test")
for d in tqdm(cwqdata):
    prompt = d['prompt']
    plan = get_model_output(prompt, tokenizer=tokenizer, model=model)
    error_cnt = 0
    while error_cnt<3 and plan[:5] != "Step1":
        plan = get_model_output(prompt, tokenizer=tokenizer, model=model)
        error_cnt +=1
    if plan[:5] != "Step1":
        print(f"unable to generate vaild plan of {d['id']}")
        print(plan)
    d['test_plan'] = plan

with open("output/cwq_test_plan.json", "w") as f:
    json.dump(cwqdata, f)