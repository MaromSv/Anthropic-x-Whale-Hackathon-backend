"""Prompt assembly and generation orchestration.

The prompt format here is the contract between this Python testbench and
the Kotlin app on-device. If you change the system prompt or context
formatting, the Kotlin team needs to mirror it exactly for the on-device
output to match what we tested.

`build_messages()` returns the EXACT message list both sides should send
to the model. Dump it to a file (`--save-prompt`) and hand it over.
"""
from __future__ import annotations

from typing import Iterable

from rag.backends import Backend, GenerationResult
from rag.retrieve import SearchHit

SYSTEM_PROMPT = (
    "You are a calm, accurate first-aid assistant for an offline crisis-response app. "
    "You will be given context snippets retrieved from medical and local Amsterdam "
    "knowledge packs. Answer the user's question using ONLY the provided context. "
    "If the context does not cover the question, say so honestly — do not invent facts. "
    "Always include: (1) the immediate physical action, (2) when to call 112, "
    "(3) any local context that matters (e.g. Dutch police amnesty for drug overdose, "
    "EHBO post locations). Keep the response under 150 words. Be direct. "
    "Lives may depend on this."
)


def _format_context(hits: Iterable[SearchHit]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h.doc.title}]  (relevance={h.score:.2f}, pack={h.doc.pack_id})\n"
            f"{h.doc.content}"
        )
    return "\n\n---\n\n".join(blocks)


def build_messages(query: str, hits: list[SearchHit]) -> list[dict]:
    """Build the message list sent to the model.

    Gemma's chat template doesn't have a 'system' role — system instructions
    go into the first user message. Both HF and llama.cpp handle this correctly
    via apply_chat_template / create_chat_completion. The Kotlin app must
    mirror this single-user-message structure to get matching outputs.
    """
    user_msg = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{_format_context(hits)}\n\n"
        f"QUESTION: {query}"
    )
    return [{"role": "user", "content": user_msg}]


def generate_answer(
    backend: Backend, query: str, hits: list[SearchHit], max_new_tokens: int = 300
) -> GenerationResult:
    messages = build_messages(query, hits)
    return backend.generate(messages, max_new_tokens=max_new_tokens)
