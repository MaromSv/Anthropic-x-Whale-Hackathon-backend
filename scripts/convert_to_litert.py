"""Convert a Gemma checkpoint (HF safetensors, optionally LoRA-fine-tuned)
into a MediaPipe `.task` bundle that the Android app loads via the
LLM Inference API.

End-to-end pipeline:

    HF safetensors  →  [merge LoRA]  →  ai_edge_torch  →  TFLite
                                                              ↓
                                  mediapipe.tasks.genai.bundler
                                                              ↓
                                                       gemma.task

Why this exists: the LiteRT package on HF (`litert-community/...`) is the
*output* of this pipeline. Once you fine-tune or re-quantize your own
Gemma, you re-run this script to produce a fresh `.task` for the Kotlin team.

CAVEATS — read these before you wonder why it's broken:
  - ai-edge-torch's Gemma authoring API has been moving fast. If the
    import paths below have shifted, check the latest examples at:
        https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative/examples
  - dtype mismatches (fp16 vs bf16) cause silent or cryptic conversion
    errors. Stick to bf16 unless you know the destination quant scheme.
  - Conversion is RAM-hungry. Gemma 2 2B needs ~16 GB free during export.
    Kaggle/Colab free tiers will OOM. Run on your Mac (32+ GB) or rent
    a beefy box for the conversion step alone.
  - If ai-edge-torch fails on your model variant, the documented fallback
    path is HF → ONNX (via `optimum-cli export onnx`) → LiteRT, but the
    resulting `.tflite` is not always compatible with MediaPipe's LLM
    Inference task. Try ai-edge-torch first.

Setup (one time, big install — ~3 GB of deps):
    pip install ai-edge-torch ai-edge-litert mediapipe>=0.10.20 \
        transformers torch peft safetensors

Then:
    python -m scripts.convert_to_litert \
        --model google/gemma-2-2b-it \
        --quantize int8 \
        --output ./build/gemma-2-2b-it.task

Optional flags:
    --lora-adapter ./my-finetune       Merge a LoRA adapter before export
    --quantize {none, int8, int4}      Default int8 (good size/quality balance)
    --max-seq-len 1024                 Context window baked into the .task
    --keep-tflite                      Also keep the intermediate .tflite file
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def merge_lora_to_base(base_id: str, adapter_path: str, out_dir: Path) -> Path:
    """Load HF base + LoRA adapter, merge weights, save full model.

    Skip this if you're converting a vanilla HF checkpoint with no fine-tuning.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[lora] loading base {base_id}...")
    base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
    print(f"[lora] applying adapter {adapter_path}...")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(base_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    print(f"[lora] merged checkpoint → {out_dir}")
    return out_dir


def convert_to_tflite(checkpoint_path: str, quantize: str, max_seq_len: int, out_path: Path) -> Path:
    """Run ai-edge-torch's Gemma authoring + TFLite conversion.

    The exact import path depends on the Gemma generation:
      - Gemma 1:    ai_edge_torch.generative.examples.gemma
      - Gemma 2/3:  ai_edge_torch.generative.examples.gemma2
    Adjust if you see ImportError on the line below.
    """
    print(f"[tflite] importing ai_edge_torch...")
    from ai_edge_torch.generative.examples.gemma2 import gemma2 as gemma_authoring
    from ai_edge_torch.generative.utilities import converter
    from ai_edge_torch.generative.layers import kv_cache as kv_utils
    from ai_edge_torch.generative.quantize import quant_recipes

    print(f"[tflite] loading checkpoint from {checkpoint_path}...")
    pytorch_model = gemma_authoring.build_2b_model(
        checkpoint_path=checkpoint_path,
        kv_cache_max_len=max_seq_len,
    )

    # Pick a quantization recipe. ai-edge-torch uses 'recipes' rather than
    # bare ints — they bundle weight scheme + activation scheme + ops.
    recipe = None
    if quantize == "int8":
        recipe = quant_recipes.full_int8_dynamic_recipe()
    elif quantize == "int4":
        recipe = quant_recipes.full_int4_recipe()
    elif quantize == "none":
        recipe = None
    else:
        raise ValueError(f"unknown quantize value: {quantize}")

    print(f"[tflite] converting (quant={quantize}, seq={max_seq_len})... this is the slow step")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    converter.convert_to_tflite(
        pytorch_model,
        output_path=str(out_path.parent),
        output_name_prefix=out_path.stem,
        prefill_seq_len=max_seq_len,
        quant_recipe=recipe,
    )
    # ai-edge-torch writes <prefix>.tflite into output_path; normalize:
    written = out_path.parent / f"{out_path.stem}.tflite"
    if written != out_path:
        written.rename(out_path)
    print(f"[tflite] wrote {out_path}")
    return out_path


def bundle_task(tflite_path: Path, tokenizer_dir: str, out_path: Path, max_seq_len: int) -> Path:
    """Wrap the .tflite + tokenizer into a MediaPipe .task bundle.

    The resulting file is what `LlmInference.createFromOptions(...)` on
    Android consumes directly. No further processing needed.
    """
    from mediapipe.tasks.python.genai import bundler

    print(f"[bundle] packing {tflite_path} + tokenizer from {tokenizer_dir}...")
    config = bundler.BundleConfig(
        tflite_model=str(tflite_path),
        tokenizer_model=str(Path(tokenizer_dir) / "tokenizer.model"),
        start_token="<bos>",
        stop_tokens=["<eos>", "<end_of_turn>"],
        output_filename=str(out_path),
        enable_bytes_to_unicode_mapping=False,
        prompt_prefix="<start_of_turn>user\n",
        prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
    )
    bundler.create_bundle(config)
    print(f"[bundle] wrote {out_path}  ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="google/gemma-2-2b-it",
                   help="HF model id OR local path to a checkpoint dir")
    p.add_argument("--lora-adapter", default=None,
                   help="optional LoRA adapter path to merge before conversion")
    p.add_argument("--quantize", choices=["none", "int8", "int4"], default="int8")
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="KV-cache size baked into the bundle (= max prompt+completion length)")
    p.add_argument("--output", default="./build/gemma.task",
                   help="path for the final .task file")
    p.add_argument("--keep-tflite", action="store_true",
                   help="don't delete the intermediate .tflite")
    args = p.parse_args()

    out_path = Path(args.output).resolve()

    # Step 1: optionally merge LoRA into a fresh checkpoint dir
    with tempfile.TemporaryDirectory(prefix="gemma_convert_") as tmp:
        tmp_dir = Path(tmp)

        if args.lora_adapter:
            ckpt_path = merge_lora_to_base(args.model, args.lora_adapter, tmp_dir / "merged")
            ckpt_path = str(ckpt_path)
        else:
            ckpt_path = args.model  # ai-edge-torch will pull from HF directly

        # Step 2: convert to .tflite
        tflite_path = tmp_dir / f"{out_path.stem}.tflite"
        convert_to_tflite(ckpt_path, args.quantize, args.max_seq_len, tflite_path)

        # Step 3: bundle .tflite + tokenizer into .task
        # We need the tokenizer dir; if user gave a HF id, snapshot-download just the tokenizer
        if Path(ckpt_path).exists():
            tokenizer_dir = str(ckpt_path)
        else:
            from huggingface_hub import snapshot_download
            print(f"[bundle] downloading tokenizer from {ckpt_path}...")
            tokenizer_dir = snapshot_download(
                ckpt_path,
                allow_patterns=["tokenizer*", "special_tokens_map.json"],
            )

        bundle_task(tflite_path, tokenizer_dir, out_path, args.max_seq_len)

        if args.keep_tflite:
            keep_at = out_path.with_suffix(".tflite")
            shutil.copy(tflite_path, keep_at)
            print(f"[bundle] kept intermediate → {keep_at}")

    print(f"\n✓ done. drop {out_path} into the Android app's assets/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
