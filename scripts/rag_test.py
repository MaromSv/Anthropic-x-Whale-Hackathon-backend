"""RAG testbench.

Three usage modes:

  1. Single query — what does this model say to this question?
        python -m scripts.rag_test query "someone fell in the canal" \
            --model hf:google/gemma-2-2b-it

  2. Eval suite — run all crisis queries, print retrieval + latency summary:
        python -m scripts.rag_test eval --model hf:google/gemma-2-2b-it
        python -m scripts.rag_test eval --no-llm   # retrieval only, fast

  3. Compare — A/B two model variants on the eval suite:
        python -m scripts.rag_test compare \
            --model hf:google/gemma-2-2b-it \
            --model gguf:./gemma-2-2b-it-Q4_K_M.gguf

  4. Benchmark — RAG vs no-RAG on the FirstAidQA dataset (the "demo slide"):
        python -m scripts.rag_test benchmark \
            --model quanto:./models/gemma-4-E2B-quanto-int4 \
            --n 30 \
            --report /tmp/rag_vs_norag.md

The point of (3) is to find the smallest acceptable quantization for the
phone. (4) gives you a measurable proof that the data packs improve answer
quality, scored against held-out reference answers.

Compatibility note: the prompt + retrieval shape here is the contract the
Kotlin app must match. Use --save-prompt to dump the exact message list
to JSON for the Android team.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

from rag.backends import Backend, build_backend
from rag.eval import EVAL_QUERIES, retrieval_score
from rag.generate import build_messages, generate_answer
from rag.retrieve import RAGIndex, SearchHit


def _print_hits(hits: list[SearchHit], indent: str = "  ") -> None:
    for h in hits:
        print(f"{indent}[{h.score:.3f}]  ({h.doc.pack_id}) {h.doc.title}")


def _load_index() -> RAGIndex:
    index = RAGIndex()
    index.build()
    return index


# --- Mode: query ------------------------------------------------------------

def cmd_query(args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    index = _load_index()
    hits = index.search(query, k=args.k, tag_filter=args.tag, pack_filter=args.pack)

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    print("\n--- retrieved ---")
    _print_hits(hits)

    if args.save_prompt:
        msgs = build_messages(query, hits)
        Path(args.save_prompt).write_text(json.dumps(msgs, indent=2, ensure_ascii=False))
        print(f"\nprompt saved → {args.save_prompt}")

    if args.no_llm or not args.model:
        return 0

    backend = build_backend(args.model)
    print(f"\n--- generating with {backend.name} ---\n")
    result = generate_answer(backend, query, hits, max_new_tokens=args.max_tokens)
    print(result.text)
    print(
        f"\n[{result.output_tokens} tok in {result.seconds:.1f}s = "
        f"{result.tokens_per_second:.1f} tok/s, prompt={result.prompt_tokens} tok]"
    )
    return 0


# --- Mode: eval -------------------------------------------------------------

def _run_eval(index: RAGIndex, backend: Optional[Backend], k: int, max_tokens: int) -> dict:
    """Run every EVAL_QUERY against the index (and optionally a backend).

    Returns a per-query record dict, suitable for printing or saving.
    """
    records = []
    print(f"\nRunning {len(EVAL_QUERIES)} eval queries"
          + (f" with {backend.name}" if backend else " (retrieval only)") + "...\n")

    for eq in EVAL_QUERIES:
        t0 = time.time()
        hits = index.search(eq.query, k=k)
        retrieval_ms = (time.time() - t0) * 1000

        hit_titles = [h.doc.title for h in hits]
        score = retrieval_score(hit_titles, eq.expected_titles)

        rec = {
            "label": eq.label,
            "query": eq.query,
            "retrieval_ms": round(retrieval_ms, 1),
            "retrieval_score": round(score, 2) if score == score else None,
            "top_hits": [{"title": h.doc.title, "score": round(h.score, 3)} for h in hits],
        }

        if backend is not None:
            r = generate_answer(backend, eq.query, hits, max_new_tokens=max_tokens)
            rec.update({
                "answer": r.text,
                "tokens": r.output_tokens,
                "seconds": round(r.seconds, 2),
                "tok_per_s": round(r.tokens_per_second, 1),
            })

        records.append(rec)

        marker = "✓" if score == 1.0 else ("~" if score and score > 0 else "✗")
        gen_info = f"  {rec.get('tok_per_s', '-')} tok/s" if backend else ""
        print(f"  {marker} {eq.label:24s}  retrieval={rec['retrieval_score']}{gen_info}")

    return {"backend": backend.name if backend else "retrieval-only", "records": records}


def cmd_eval(args: argparse.Namespace) -> int:
    index = _load_index()
    backend = None if args.no_llm else build_backend(args.model)
    result = _run_eval(index, backend, k=args.k, max_tokens=args.max_tokens)

    if args.save_to:
        Path(args.save_to).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nresults saved → {args.save_to}")

    # Quick aggregate
    scores = [r["retrieval_score"] for r in result["records"] if r["retrieval_score"] is not None]
    avg = sum(scores) / len(scores) if scores else 0
    print(f"\nAvg retrieval score: {avg:.2f} (1.0 = all expected titles in top-{args.k})")
    if backend:
        tps = [r["tok_per_s"] for r in result["records"] if "tok_per_s" in r]
        if tps:
            print(f"Avg generation:      {sum(tps)/len(tps):.1f} tok/s")
    return 0


# --- Mode: compare ----------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    if len(args.model) < 2:
        print("compare needs at least two --model specs")
        return 1

    index = _load_index()
    all_results = []

    for spec in args.model:
        backend = build_backend(spec)
        result = _run_eval(index, backend, k=args.k, max_tokens=args.max_tokens)
        all_results.append(result)
        # free GPU/MPS memory before loading the next backend
        del backend

    # Side-by-side table
    print("\n" + "=" * 90)
    print(f"{'Query':<26}" + "".join(f"{r['backend'][:28]:<32}" for r in all_results))
    print("=" * 90)
    for i, eq in enumerate(EVAL_QUERIES):
        row = f"{eq.label:<26}"
        for r in all_results:
            rec = r["records"][i]
            tps = rec.get("tok_per_s", "-")
            sec = rec.get("seconds", "-")
            row += f"{tps} tok/s ({sec}s){'':<10}"[:32]
        print(row)

    if args.save_to:
        Path(args.save_to).write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"\nresults saved → {args.save_to}")
    return 0


# --- CLI plumbing -----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_test", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--k", type=int, default=5)
    common.add_argument("--max-tokens", type=int, default=300)

    pq = sub.add_parser("query", parents=[common], help="single query against one model")
    pq.add_argument("query", nargs="+")
    pq.add_argument("--model", default=None, help="hf:<id> or gguf:<path>; omit to skip generation")
    pq.add_argument("--tag", default=None)
    pq.add_argument("--pack", default=None)
    pq.add_argument("--no-llm", action="store_true")
    pq.add_argument("--save-prompt", default=None,
                    help="write the exact message list to JSON (for the Kotlin team)")
    pq.set_defaults(func=cmd_query)

    pe = sub.add_parser("eval", parents=[common], help="run the full crisis eval suite")
    pe.add_argument("--model", default="hf:google/gemma-2-2b-it")
    pe.add_argument("--no-llm", action="store_true",
                    help="retrieval only — fast, useful for tuning the index")
    pe.add_argument("--save-to", default=None, help="path to write results JSON")
    pe.set_defaults(func=cmd_eval)

    pc = sub.add_parser("compare", parents=[common],
                        help="A/B two or more model variants on the eval suite")
    pc.add_argument("--model", action="append", default=[],
                    help="repeatable; e.g. --model hf:google/gemma-2-2b-it --model gguf:foo.gguf")
    pc.add_argument("--save-to", default=None)
    pc.set_defaults(func=cmd_compare)

    pb = sub.add_parser("benchmark", parents=[common],
                        help="RAG vs no-RAG on FirstAidQA — quality benchmark for the demo")
    pb.add_argument("--model", required=True)
    pb.add_argument("--n", type=int, default=30, help="number of questions to sample")
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--report", default=None, help="markdown report path (side-by-side comparison)")
    pb.add_argument("--save-to", default=None, help="raw JSON results path")
    pb.set_defaults(func=cmd_benchmark)

    return p


# --- Mode: benchmark --------------------------------------------------------

def cmd_benchmark(args: argparse.Namespace) -> int:
    from dataclasses import asdict
    from rag.benchmark import run_benchmark, write_markdown_report

    index = _load_index()
    backend = build_backend(args.model)
    records = run_benchmark(
        backend, index,
        n_questions=args.n, k=args.k,
        max_new_tokens=args.max_tokens, seed=args.seed,
    )

    # Aggregate
    import statistics
    rag_sims = [r.rag_similarity for r in records]
    no_rag_sims = [r.no_rag_similarity for r in records]
    rag_wins = sum(1 for r in records if r.rag_similarity > r.no_rag_similarity)

    print("\n" + "=" * 60)
    print(f"{'Backend:':<12} {backend.name}")
    print(f"{'Questions:':<12} {len(records)}")
    print(f"{'RAG sim:':<12} {statistics.mean(rag_sims):.3f}  (median {statistics.median(rag_sims):.3f})")
    print(f"{'no-RAG sim:':<12} {statistics.mean(no_rag_sims):.3f}  (median {statistics.median(no_rag_sims):.3f})")
    print(f"{'RAG wins:':<12} {rag_wins}/{len(records)}  ({100*rag_wins/len(records):.0f}%)")
    print("=" * 60)

    if args.save_to:
        Path(args.save_to).write_text(json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False))
        print(f"raw JSON  → {args.save_to}")
    if args.report:
        write_markdown_report(records, Path(args.report))
        print(f"markdown  → {args.report}")
    return 0


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
