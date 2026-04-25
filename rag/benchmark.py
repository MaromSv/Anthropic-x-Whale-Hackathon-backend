"""RAG vs no-RAG benchmark on the FirstAidQA dataset.

Pulls N questions from `i-am-mushfiq/FirstAidQA` and asks the model each
question twice — once with retrieved context from our packs (RAG), once
without (no-RAG). Both answers are compared to the dataset's reference
answer via cosine similarity in embedding space.

Output:
  - Per-question record: question, reference, both answers, similarities
  - Aggregate: mean similarity (RAG vs no-RAG), latency, win rate
  - Markdown report for the demo (eyeballable side-by-side)

The point: prove the data packs measurably improve answer quality on a
held-out, realistic Q&A distribution. This is the backbone of any
"with RAG vs without" demo slide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from rag.backends import Backend, GenerationResult
from rag.generate import SYSTEM_PROMPT, build_messages, generate_answer
from rag.retrieve import RAGIndex


@dataclass
class BenchmarkRecord:
    question: str
    reference: str
    rag_answer: str
    rag_seconds: float
    rag_tok_per_s: float
    rag_similarity: float
    rag_top_titles: list[str] = field(default_factory=list)
    no_rag_answer: str = ""
    no_rag_seconds: float = 0.0
    no_rag_tok_per_s: float = 0.0
    no_rag_similarity: float = 0.0


# A "no-RAG" message: same system prompt, no context block.
# Fair comparison — only the retrieval is varied.
def _no_rag_messages(question: str) -> list[dict]:
    user_msg = (
        f"{SYSTEM_PROMPT}\n\n"
        f"(No retrieved context is available — answer from your own knowledge.)\n\n"
        f"QUESTION: {question}"
    )
    return [{"role": "user", "content": user_msg}]


def _embed(texts: list[str], embed_model) -> np.ndarray:
    return embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def run_benchmark(
    backend: Backend,
    index: RAGIndex,
    n_questions: int = 30,
    k: int = 5,
    max_new_tokens: int = 250,
    seed: int = 42,
) -> list[BenchmarkRecord]:
    from datasets import load_dataset
    ds = load_dataset("i-am-mushfiq/FirstAidQA", split="train")
    ds = ds.shuffle(seed=seed).select(range(n_questions))

    # Reuse the embedding model from the index for similarity scoring
    embed_model = index._model

    records: list[BenchmarkRecord] = []
    for i, row in enumerate(ds):
        q = row["question"]
        ref = row["answer"]
        print(f"\n[{i+1}/{n_questions}] {q}")

        # --- RAG path ---
        hits = index.search(q, k=k)
        rag_result = generate_answer(backend, q, hits, max_new_tokens=max_new_tokens)

        # --- No-RAG path (same backend, no context) ---
        no_rag_result = backend.generate(_no_rag_messages(q), max_new_tokens=max_new_tokens)

        # --- Similarity scoring (cosine vs reference) ---
        embs = _embed([ref, rag_result.text, no_rag_result.text], embed_model)
        rag_sim = float(np.dot(embs[0], embs[1]))
        no_rag_sim = float(np.dot(embs[0], embs[2]))

        record = BenchmarkRecord(
            question=q,
            reference=ref,
            rag_answer=rag_result.text,
            rag_seconds=rag_result.seconds,
            rag_tok_per_s=rag_result.tokens_per_second,
            rag_similarity=rag_sim,
            rag_top_titles=[h.doc.title for h in hits],
            no_rag_answer=no_rag_result.text,
            no_rag_seconds=no_rag_result.seconds,
            no_rag_tok_per_s=no_rag_result.tokens_per_second,
            no_rag_similarity=no_rag_sim,
        )
        records.append(record)

        winner = "RAG" if rag_sim > no_rag_sim else "no-RAG" if no_rag_sim > rag_sim else "tie"
        print(f"   RAG sim={rag_sim:.3f}   no-RAG sim={no_rag_sim:.3f}   → {winner}")

    return records


def write_markdown_report(records: list[BenchmarkRecord], out_path: Path) -> None:
    """A demo-friendly side-by-side comparison."""
    rag_sims = [r.rag_similarity for r in records]
    no_rag_sims = [r.no_rag_similarity for r in records]
    rag_wins = sum(1 for r in records if r.rag_similarity > r.no_rag_similarity)
    no_rag_wins = sum(1 for r in records if r.no_rag_similarity > r.rag_similarity)

    lines = [
        "# FirstAidQA Benchmark — RAG vs no-RAG\n",
        f"**Dataset:** [`i-am-mushfiq/FirstAidQA`](https://huggingface.co/datasets/i-am-mushfiq/FirstAidQA)  ",
        f"**Questions:** {len(records)}  ",
        f"**Metric:** cosine similarity to reference answer (MiniLM embeddings)\n",
        "## Aggregate\n",
        f"| | mean sim | wins | mean tok/s |",
        f"|---|---|---|---|",
        f"| **RAG**     | {np.mean(rag_sims):.3f}    | {rag_wins:2d} / {len(records)} | {np.mean([r.rag_tok_per_s for r in records]):.1f} |",
        f"| **no-RAG**  | {np.mean(no_rag_sims):.3f} | {no_rag_wins:2d} / {len(records)} | {np.mean([r.no_rag_tok_per_s for r in records]):.1f} |",
        f"| **delta**   | {np.mean(rag_sims) - np.mean(no_rag_sims):+.3f} | | |\n",
        "## Per-question\n",
    ]
    for i, r in enumerate(records, 1):
        winner = "🟢 RAG" if r.rag_similarity > r.no_rag_similarity else "🔴 no-RAG" if r.no_rag_similarity > r.rag_similarity else "🟡 tie"
        lines += [
            f"### {i}. {r.question}",
            f"**Winner:** {winner}  (RAG={r.rag_similarity:.3f}, no-RAG={r.no_rag_similarity:.3f})  ",
            f"**Retrieved:** {', '.join(r.rag_top_titles[:3])}\n",
            f"<details><summary>Reference / RAG / no-RAG answers</summary>\n",
            f"**Reference:** {r.reference}\n",
            f"**RAG answer:** {r.rag_answer}\n",
            f"**no-RAG answer:** {r.no_rag_answer}\n",
            "</details>\n",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
