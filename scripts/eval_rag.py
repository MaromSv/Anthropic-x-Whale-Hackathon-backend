"""Evaluate RAG vs no-RAG on FirstAidQA samples.

Loads the model, builds the RAG index from the existing data packs, then runs
each question twice — once with retrieved context (RAG) and once without.
Scores both answers against the dataset reference via cosine similarity.

Works with both the full HF model and a Quanto-quantized model (auto-detected).

Usage:
    python scripts/eval_rag.py --model ./models/gemma-4-E2B-it
    python scripts/eval_rag.py --model ./models/gemma-4-E2B-it-quanto-int4

Options:
    --model       Model path (required; quanto auto-detected via quanto_qmap.json)
    --n           Questions to sample (default: 20)
    --k           Retrieved docs per question (default: 5)
    --seed        Dataset shuffle seed (default: 42)
    --max-tokens  Max new tokens per answer (default: 200)
    --output      Optional path to write JSON results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from rag.backends import HFBackend, QuantoBackend
from rag.generate import SYSTEM_PROMPT, generate_answer
from rag.retrieve import EMBEDDING_MODEL, RAGIndex


def _load_backend(model_path: str):
    if (Path(model_path) / "quanto_qmap.json").exists():
        print(f"Detected Quanto-quantized model")
        return QuantoBackend(model_path)
    print(f"Detected full HF model")
    return HFBackend(model_path)


def _no_rag_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": (
        f"{SYSTEM_PROMPT}\n\n"
        f"(No retrieved context available — answer from your own knowledge.)\n\n"
        f"QUESTION: {question}"
    )}]


def _embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def main() -> int:
    p = argparse.ArgumentParser(description="RAG vs no-RAG eval on FirstAidQA")
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--k", type=int, default=5, help="Retrieved docs per question")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-tokens", type=int, default=200, dest="max_tokens")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    # ── Build RAG index ────────────────────────────────────────────────────
    index = RAGIndex()
    index.build()
    embedder = index._model  # reuse the same embedding model for scoring

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model}...")
    backend = _load_backend(args.model)

    # ── Load dataset ───────────────────────────────────────────────────────
    print(f"\nLoading FirstAidQA ({args.n} samples, seed={args.seed})...")
    ds = load_dataset("i-am-mushfiq/FirstAidQA", split="train")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    # ── Run eval ───────────────────────────────────────────────────────────
    records: list[dict] = []
    rag_wins = no_rag_wins = 0

    col_q = 45
    print(f"\n{'#':>3}  {'RAG':>5}  {'noRAG':>5}  {'Δ':>5}  {'tok/s':>5}  Question")
    print("─" * (3 + 2 + 5 + 2 + 5 + 2 + 5 + 2 + 5 + 2 + col_q))

    for i, row in enumerate(ds):
        q: str = row["question"]
        ref: str = row["answer"]

        hits = index.search(q, k=args.k)
        rag_result = generate_answer(backend, q, hits, max_new_tokens=args.max_tokens)
        no_rag_result = backend.generate(_no_rag_messages(q), max_new_tokens=args.max_tokens)

        embs = _embed([ref, rag_result.text, no_rag_result.text], embedder)
        rag_sim = float(np.dot(embs[0], embs[1]))
        no_rag_sim = float(np.dot(embs[0], embs[2]))
        delta = rag_sim - no_rag_sim

        if rag_sim > no_rag_sim:
            rag_wins += 1
        elif no_rag_sim > rag_sim:
            no_rag_wins += 1

        print(f"{i+1:>3}  {rag_sim:>5.3f}  {no_rag_sim:>5.3f}  {delta:>+5.3f}  "
              f"{rag_result.tokens_per_second:>5.1f}  {q[:col_q]}")

        records.append({
            "question": q,
            "reference": ref,
            "rag_answer": rag_result.text,
            "no_rag_answer": no_rag_result.text,
            "rag_similarity": rag_sim,
            "no_rag_similarity": no_rag_sim,
            "retrieved_titles": [h.doc.title for h in hits],
            "rag_tok_per_s": rag_result.tokens_per_second,
            "no_rag_tok_per_s": no_rag_result.tokens_per_second,
        })

    # ── Summary ────────────────────────────────────────────────────────────
    rag_sims = [r["rag_similarity"] for r in records]
    no_rag_sims = [r["no_rag_similarity"] for r in records]
    n = len(records)

    print("─" * (3 + 2 + 5 + 2 + 5 + 2 + 5 + 2 + 5 + 2 + col_q))
    print(f"\n{'':5} {'mean sim':>10}  {'wins':>6}  {'mean tok/s':>10}")
    print(f"{'RAG':5} {np.mean(rag_sims):>10.4f}  {rag_wins:>3}/{n}  "
          f"{np.mean([r['rag_tok_per_s'] for r in records]):>10.1f}")
    print(f"{'noRAG':5} {np.mean(no_rag_sims):>10.4f}  {no_rag_wins:>3}/{n}  "
          f"{np.mean([r['no_rag_tok_per_s'] for r in records]):>10.1f}")
    print(f"{'delta':5} {np.mean(rag_sims) - np.mean(no_rag_sims):>+10.4f}")
    print(f"\nModel : {args.model}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": args.model,
            "n_questions": n,
            "k": args.k,
            "seed": args.seed,
            "rag_mean_similarity": float(np.mean(rag_sims)),
            "no_rag_mean_similarity": float(np.mean(no_rag_sims)),
            "delta": float(np.mean(rag_sims) - np.mean(no_rag_sims)),
            "rag_wins": rag_wins,
            "no_rag_wins": no_rag_wins,
            "records": records,
        }, indent=2))
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
