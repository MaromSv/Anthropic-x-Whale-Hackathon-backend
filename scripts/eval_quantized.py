"""Evaluate a Quanto-quantized model on samples from the FirstAidQA dataset.

No RAG, no retrieval — just direct Q&A against the quantized model so you
can measure answer quality and generation speed in isolation.

Usage:
    python scripts/eval_quantized.py --model ./models/gemma-4-E2B-it-quanto-int4

Options:
    --model       Path to Quanto-quantized model directory (required)
    --n           Number of questions to sample (default: 20)
    --seed        RNG seed for reproducible sampling (default: 42)
    --max-tokens  Max new tokens per answer (default: 200)
    --output      Optional path to write JSON results (e.g. results/eval.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `rag` package is importable when this
# script is run directly (e.g. `python scripts/eval_quantized.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from rag.backends import HFBackend, QuantoBackend

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_backend(model_path: str):
    """Auto-detect: use QuantoBackend if quanto_qmap.json exists, else HFBackend."""
    if (Path(model_path) / "quanto_qmap.json").exists():
        print(f"Detected Quanto-quantized model at {model_path}")
        return QuantoBackend(model_path)
    print(f"Detected full HF model at {model_path}")
    return HFBackend(model_path)


def _embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def main() -> int:
    p = argparse.ArgumentParser(description="Eval quantized model on FirstAidQA samples")
    p.add_argument("--model", required=True, help="Path to Quanto-quantized model dir")
    p.add_argument("--n", type=int, default=20, help="Number of questions (default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Dataset shuffle seed (default: 42)")
    p.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    p.add_argument("--output", default=None, help="Optional JSON output path")
    p.add_argument("--show-prompt", action="store_true",
                   help="Print the formatted prompt for the first question (debug)")
    args = p.parse_args()

    print(f"Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    print(f"Loading model from {args.model}...")
    backend = _load_backend(args.model)

    print(f"Loading FirstAidQA ({args.n} samples, seed={args.seed})...")
    ds = load_dataset("i-am-mushfiq/FirstAidQA", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    records: list[dict] = []
    col_q = 55  # truncate question for table display

    print(f"\n{'#':>3}  {'Sim':>5}  {'Tok/s':>6}  Question")
    print("─" * (3 + 2 + 5 + 2 + 6 + 2 + col_q))

    for i, row in enumerate(ds):
        q: str = row["question"]
        ref: str = row["answer"]

        messages = [{"role": "user", "content": q}]
        if i == 0 and args.show_prompt:
            from rag.backends import _format_prompt
            print("\n--- PROMPT (first question) ---")
            print(repr(_format_prompt(backend.tokenizer, messages)))
            print("--- END PROMPT ---\n")

        result = backend.generate(messages, max_new_tokens=args.max_tokens)

        embs = _embed([ref, result.text], embedder)
        sim = float(np.dot(embs[0], embs[1]))

        print(f"{i+1:>3}  {sim:>5.3f}  {result.tokens_per_second:>6.1f}  {q[:col_q]}")

        records.append({
            "question": q,
            "reference": ref,
            "answer": result.text,
            "similarity": sim,
            "output_tokens": result.output_tokens,
            "tokens_per_second": result.tokens_per_second,
            "seconds": result.seconds,
        })

    sims = [r["similarity"] for r in records]
    toks = [r["tokens_per_second"] for r in records]

    print("─" * (3 + 2 + 5 + 2 + 6 + 2 + col_q))
    print(f"{'AVG':>3}  {np.mean(sims):>5.3f}  {np.mean(toks):>6.1f}")
    print()
    print(f"Model            : {args.model}")
    print(f"Questions        : {len(records)}")
    print(f"Mean similarity  : {np.mean(sims):.4f}  (std {np.std(sims):.4f})")
    print(f"Mean tok/s       : {np.mean(toks):.1f}")
    print(f"Total wall time  : {sum(r['seconds'] for r in records):.1f}s")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": args.model,
            "n_questions": len(records),
            "seed": args.seed,
            "mean_similarity": float(np.mean(sims)),
            "std_similarity": float(np.std(sims)),
            "mean_tok_per_s": float(np.mean(toks)),
            "records": records,
        }, indent=2))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
