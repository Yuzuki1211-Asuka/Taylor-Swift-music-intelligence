import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None
_model_dir = None


def find_qwen_model_dir():
    """
    Model loading priority:
    1. Use QWEN_MODEL_DIR environment variable if provided.
    2. Search common local model cache directories.
    """

    env_model_dir = os.environ.get("QWEN_MODEL_DIR")

    if env_model_dir:
        config_path = os.path.join(env_model_dir, "config.json")
        if os.path.exists(config_path):
            return env_model_dir

        raise FileNotFoundError(
            f"QWEN_MODEL_DIR is set but config.json was not found: {env_model_dir}"
        )

    search_roots = [
        "/root/autodl-tmp/models",
        "./models",
        "../models",
    ]

    matches = []

    for root in search_roots:
        matches.extend(
            glob.glob(
                f"{root}/**/Qwen*Qwen2.5-3B-Instruct*/snapshots/*/config.json",
                recursive=True,
            )
        )

        matches.extend(
            glob.glob(
                f"{root}/**/Qwen--Qwen2.5-3B-Instruct*/snapshots/*/config.json",
                recursive=True,
            )
        )

    if not matches:
        raise FileNotFoundError(
            "没有找到 Qwen 模型 config.json。请设置环境变量 QWEN_MODEL_DIR 指向本地模型目录。"
        )

    return os.path.dirname(matches[0])


def get_qwen():
    global _tokenizer, _model, _model_dir

    if _tokenizer is None or _model is None:
        _model_dir = find_qwen_model_dir()
        print(f"Loading Qwen from: {_model_dir}")

        _tokenizer = AutoTokenizer.from_pretrained(
            _model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            _model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )

        _model.eval()

    return _tokenizer, _model


def qwen_chat(
    user_prompt: str,
    system_prompt: str = "你是一个严谨的数据分析助手。",
    max_new_tokens: int = 700,
):
    tokenizer, model = get_qwen()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.8,
            repetition_penalty=1.05,
            do_sample=True,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()
