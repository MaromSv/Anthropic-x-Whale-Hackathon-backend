"""Quantize an HF Gemma model with Quanto (pure-PyTorch, architecture-agnostic).

Why Quanto and not MLX/llama.cpp/litert-torch?
  Those tools each maintain a per-architecture re-implementation. Gemma 4
  E2B's MatFormer pattern (different attention shapes in layers 0-14 vs
  15-34) isn't in any of them yet. Quanto operates on nn.Linear modules
  generically — load the model normally via transformers, then Quanto
  walks the module tree and replaces every Linear with a quantized one.
  No architecture-specific code needed.

Tradeoff: Quanto-quantized models stay in PyTorch land. They run on Mac
via MPS/CPU, but they don't go on Android directly — the on-device app
still uses the prebuilt `litert-community/...` `.task` file. This script
exists to:
  1. See how aggressively we can quantize before quality breaks
  2. Test the quantized model with rag_test.py to find the floor
  3. Have a fast(er), smaller model for backend dev iteration

Usage (run from repo root):
    python -m scripts.quantize_quanto \\
        --model ./models/gemma-4-E2B \\
        --output ./models/gemma-4-E2B-quanto-int4 \\
        --bits int4

Options:
    --bits {int8, int4, int2}   Default int4. int2 often produces gibberish
                                 for instruct models, but try it for the
                                 'smallest possible' demo angle.

RAM: peaks around 2x the FP16 model size during quantization. For Gemma 4
E2B (~5 GB FP16) expect ~10-12 GB peak. Should fit on 16 GB Macs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from optimum.quanto import freeze, qint2, qint4, qint8, quantization_map, quantize
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BIT_TO_QTYPE = {
    "int8": qint8,
    "int4": qint4,
    "int2": qint2,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to HF model dir or HF id")
    p.add_argument("--output", required=True, help="Where to save the quantized model")
    p.add_argument("--bits", choices=list(BIT_TO_QTYPE), default="int4")
    p.add_argument("--device", default="cpu",
                   help="Device for quantization (cpu safest; mps risks OOM)")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] loading {args.model} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      loaded {n_params/1e9:.2f}B params")

    qtype = BIT_TO_QTYPE[args.bits]
    print(f"[2/4] quantizing weights → {args.bits} (this is the slow step)...")
    quantize(model, weights=qtype)
    # `freeze` materializes the quantized tensors so save_pretrained writes
    # the actual int weights, not the original FP16 + dynamic quant calls.
    freeze(model)
    print(f"      quantization complete")

    print(f"[3/4] saving to {out_dir}...")
    # transformers `save_pretrained` mostly works with quanto-frozen modules,
    # but the safest persistence path is: save the underlying tensors via
    # safetensors + the quantization map, then reload with quanto's helper.
    state_dict = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out_dir / "model.safetensors"))

    qmap = quantization_map(model)
    (out_dir / "quanto_qmap.json").write_text(__import__("json").dumps(qmap, indent=2))

    # Also save config + tokenizer so the dir is self-contained
    model.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Disk size summary
    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    print(f"[4/4] done. directory size: {total/1e9:.2f} GB ({total/1e6:.0f} MB)")
    print(f"\nLoad it with:")
    print(f"    python -m scripts.rag_test eval --model quanto:{out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
