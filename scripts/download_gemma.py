"""Download a Gemma model from Hugging Face.

Defaults to google/gemma-3-270m (the smallest Gemma, 270M params ~200MB).
Downloads to ./models/gemma-3-270m/ by default.

NOTE: Gemma models are gated — you must:
  1. Go to https://huggingface.co/google/gemma-3-270m
  2. Accept the license/agreement
  3. Create a token at https://huggingface.co/settings/tokens
  4. Run: huggingface-cli login  (or set HF_TOKEN env var)

Usage:
    python scripts/download_gemma.py                               # default: gemma-3-270m
    python scripts/download_gemma.py --model google/gemma-2-2b     # specify another
    python scripts/download_gemma.py --output ./my_models/         # custom output dir
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a Gemma model from Hugging Face")
    parser.add_argument(
        "--model",
        default="google/gemma-3-270m",
        help="Hugging Face model repo (default: google/gemma-3-270m)",
    )
    parser.add_argument(
        "--output",
        default="./models",
        help="Output parent directory (default: ./models)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model.split("/")[-1]
    local_dir = output_dir / model_name

    print(f"Downloading {args.model} → {local_dir}")
    print("This may take a while (model files are ~1-5 GB)...")

    snapshot_download(
        repo_id=args.model,
        local_dir=str(local_dir),
    )

    # Show what we got
    total_size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
    files = list(local_dir.rglob("*"))
    print(f"\nDone! Downloaded {len(files)} files ({total_size / 1024**3:.2f} GB)")
    print(f"Model saved to: {local_dir}")


if __name__ == "__main__":
    main()
