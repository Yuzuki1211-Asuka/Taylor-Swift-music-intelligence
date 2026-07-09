import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

matches = glob.glob(
    "/root/autodl-tmp/models/**/Qwen*Qwen2.5-3B-Instruct*/snapshots/*/config.json",
    recursive=True,
)

if not matches:
    matches = glob.glob(
        "/root/autodl-tmp/models/**/config.json",
        recursive=True,
    )

print("Found config files:")
for m in matches:
    print(" -", m)

if not matches:
    raise FileNotFoundError("没有找到 config.json，请确认模型是否下载完成。")

MODEL_DIR = os.path.dirname(matches[0])
print("Using MODEL_DIR:", MODEL_DIR)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
)

model.eval()

messages = [
    {"role": "system", "content": "你是一个严谨的数据分析助手。"},
    {"role": "user", "content": "用一句话解释什么是 RAG。"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.3,
        top_p=0.8,
        repetition_penalty=1.05,
        do_sample=True,
    )

generated = outputs[0][inputs.input_ids.shape[1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))
