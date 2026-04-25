"""Fixed evaluation suite for the RAG testbench.

These are the queries we use to compare model variants and quantization
levels. Edit the list as the demo evolves — the goal is realistic phrasing
a panicked tourist on King's Day might actually type or speak.

`expected_tags` and `expected_titles` are loose oracles: if retrieval doesn't
surface a doc with one of those tags / titles in the top-K, retrieval is
broken regardless of how good the model is.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalQuery:
    query: str
    label: str
    expected_tags: list[str] = field(default_factory=list)
    expected_titles: list[str] = field(default_factory=list)


EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery(
        query="my friend took GHB an hour ago and won't wake up, his breathing is really slow",
        label="ghb_overdose",
        expected_tags=["kings_day_priority"],
        expected_titles=["GHB toxicity", "Opioid overdose"],
    ),
    EvalQuery(
        query="someone fell into the canal at Prinsengracht, what do I do",
        label="canal_fall",
        expected_tags=["canal", "drowning"],
        expected_titles=["Drowning", "Canal Water Rescue — King's Day Context"],
    ),
    EvalQuery(
        query="my mate took too much MDMA and is overheating and confused",
        label="mdma_overheat",
        expected_tags=["kings_day_priority"],
        expected_titles=["Ecstasy (MDMA) toxicity", "Heat stroke"],
    ),
    EvalQuery(
        query="will I get arrested if I call 112 for a drug overdose in Amsterdam",
        label="amnesty_question",
        expected_tags=["amnesty", "legal"],
        expected_titles=["Drug Overdose — Police Amnesty (Netherlands)"],
    ),
    EvalQuery(
        query="where is the nearest first aid post on Kings Day",
        label="ehbo_location",
        expected_tags=["ehbo"],
        expected_titles=[
            "EHBO First-Aid Posts — King's Day Amsterdam (Official)",
            "Non-Emergency First Aid — Use EHBO, Not 112",
        ],
    ),
    EvalQuery(
        query="someone is having a seizure",
        label="seizure",
        expected_titles=["Seizure", "Status epilepticus"],
    ),
    EvalQuery(
        query="my friend is unconscious but breathing after drinking way too much",
        label="alcohol_unconscious",
        expected_tags=["kings_day_priority"],
        expected_titles=["Alcohol intoxication"],
    ),
    EvalQuery(
        query="he's bleeding heavily from a cut on his leg from broken glass",
        label="bleeding_glass",
        expected_titles=["Hemorrhage", "Laceration repair", "Tourniquet"],
    ),
    EvalQuery(
        query="someone collapsed and isn't breathing",
        label="cardiac_arrest",
        expected_titles=["Cardiac arrest", "CPR"],
    ),
    EvalQuery(
        query="help, I think I'm having a panic attack in this crowd",
        label="panic_attack",
        expected_titles=["Panic attack"],
    ),
]


def retrieval_score(hit_titles: list[str], expected_titles: list[str]) -> float:
    """Crude oracle: fraction of expected titles found in retrieved set."""
    if not expected_titles:
        return float("nan")
    found = sum(1 for t in expected_titles if t in hit_titles)
    return found / len(expected_titles)
