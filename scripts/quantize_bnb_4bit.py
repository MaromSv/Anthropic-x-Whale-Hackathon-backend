"""Quantize Gemma-4 in 4-bit (NF4) with bitsandbytes and save locally.

This script is intended for GPU nodes (e.g., H100) where CUDA is available.

Example:
    python -m scripts.quantize_bnb_4bit \
        --model ./models/gemma-4-E2B-it \
        --output ./models/gemma-4-E2B-it-bnb-nf4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4-bit NF4 quantization for Gemma-4")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Smoke-test generation length")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for bitsandbytes 4-bit quantization")

    print(f"[1/4] loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("[2/4] loading model with bitsandbytes NF4...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    print(f"[3/4] saving quantized checkpoint to {out_dir}...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    print("[4/4] smoke-test generation...")
    messages = [{"role": "user", "content": "What is first aid?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True).strip()
    print("Smoke output:", text[:300])

    (out_dir / "quantization_info.json").write_text(json.dumps({
        "quantization": "bitsandbytes-nf4",
        "base_model": args.model,
        "dtype": "bfloat16",
    }, indent=2))

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
