"""One-time builder for the Core Medical pack.

Run from the repo root:
    python -m scripts.build_core_medical

Downloads a hand-curated whitelist of ~80 high-impact emergency-medicine
topics from WikEM, tags King's-Day-relevant entries, and writes
`data/core_medical.json`. The FastAPI route serves that file directly — no
live scraping at request time.

Why a whitelist (not a full category dump)? RAG quality > quantity.
The previous category-dump approach pulled 1148 pages including ICU
protocols, vasopressor management, and other clinical content irrelevant
to a tourist with a phone. Fewer high-signal docs retrieve better.
"""
import json
import sys
import time
from pathlib import Path

from packs.tags import kings_day_tags_for
from scrapers.wikem import fetch_wikem_page, page_url

# --- Curated topic list ---
# Each entry: WikEM page title → our `category` label for the RAGDocument.
# WikEM has redirects on, so close-but-not-exact titles usually resolve.
# Failures are skipped — review them in the run summary and adjust.

ESSENTIAL_TITLES: dict[str, str] = {
    # --- Cardiac ---
    "CPR": "Cardiac",
    "Cardiac arrest": "Cardiac",
    "Acute coronary syndrome": "Cardiac",
    "Myocardial infarction": "Cardiac",
    "Atrial fibrillation": "Cardiac",
    "Syncope": "Cardiac",
    "Bradycardia": "Cardiac",
    "Tachycardia": "Cardiac",
    "Hypertensive emergency": "Cardiac",
    "Pulmonary embolism": "Cardiac",

    # --- Trauma ---
    "Trauma": "Trauma",
    "Hemorrhage": "Trauma",
    "Tourniquet": "Trauma",
    "Head injury": "Trauma",
    "Concussion": "Trauma",
    "Cervical spine injury": "Trauma",
    "Burn": "Trauma",
    "Laceration repair": "Trauma",
    "Fracture": "Trauma",
    "Dislocation": "Trauma",
    "Penetrating trauma": "Trauma",
    "Dental trauma": "Trauma",
    "Eye trauma": "Trauma",
    "Crush injury": "Trauma",
    "Nosebleed": "Trauma",

    # --- Toxicology (King's Day hotlist + common drugs) ---
    "Alcohol intoxication": "Toxicology",
    "Ecstasy (MDMA) toxicity": "Toxicology",
    "GHB toxicity": "Toxicology",
    "Cocaine toxicity": "Toxicology",
    "Amphetamine toxicity": "Toxicology",
    "Nitrous oxide toxicity": "Toxicology",
    "Opioid overdose": "Toxicology",
    "Benzodiazepine toxicity": "Toxicology",
    "Naloxone": "Toxicology",
    "Acetaminophen toxicity": "Toxicology",
    "Cannabis toxicity": "Toxicology",
    "Ketamine toxicity": "Toxicology",
    "LSD toxicity": "Toxicology",
    "Mushroom toxicity": "Toxicology",
    "Carbon monoxide toxicity": "Toxicology",
    "Serotonin syndrome": "Toxicology",
    "Aspirin toxicity": "Toxicology",

    # --- Environmental ---
    "Hypothermia": "Environmental",
    "Frostbite": "Environmental",
    "Heat stroke": "Environmental",
    "Heat exhaustion": "Environmental",
    "Drowning": "Environmental",
    "Lightning injury": "Environmental",
    "Electrical injury": "Environmental",
    "Snake bite": "Environmental",
    "Jellyfish envenomation": "Environmental",

    # --- Allergy ---
    "Anaphylaxis": "Allergy",
    "Angioedema": "Allergy",
    "Allergic reaction": "Allergy",

    # --- Neurology ---
    "Stroke": "Neurology",
    "Seizure": "Neurology",
    "Status epilepticus": "Neurology",
    "Headache": "Neurology",
    "Subarachnoid hemorrhage": "Neurology",
    "Migraine": "Neurology",

    # --- Respiratory ---
    "Choking": "Respiratory",
    "Asthma exacerbation": "Respiratory",
    "COPD exacerbation": "Respiratory",
    "Pneumothorax": "Respiratory",
    "Hyperventilation": "Respiratory",
    "Pulmonary edema": "Respiratory",

    # --- Psychiatry / Behavioral ---
    "Agitated or combative patient": "Psychiatry",
    "Panic attack": "Psychiatry",
    "Suicidal ideation": "Psychiatry",
    "Psychosis": "Psychiatry",

    # --- General / EMS ---
    "Shock": "EMS",
    "Sepsis": "EMS",
    "Hypoglycemia": "EMS",
    "Hyperglycemia": "EMS",
    "Diabetic ketoacidosis": "EMS",
    "Dehydration": "EMS",
    "Acute abdomen": "EMS",
    "Gastrointestinal bleed": "EMS",
    "Chest pain": "EMS",
    "Abdominal pain": "EMS",
    "Back pain": "EMS",
    "Vomiting": "EMS",
    "Diarrhea": "EMS",
    "Fever": "EMS",
    "Vaginal bleeding": "EMS",
}

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "core_medical.json"
SLEEP_BETWEEN_FETCHES = 0.2
MIN_CONTENT_CHARS = 200


def build() -> None:
    print("=== Building Core Medical pack ===\n")
    print(f"Curated whitelist: {len(ESSENTIAL_TITLES)} titles")
    print(f"Estimated time: ~{len(ESSENTIAL_TITLES) * 1.5 / 60:.1f} minutes\n")

    documents: list[dict] = []
    failures: list[tuple[str, str]] = []

    for i, (title, cat_label) in enumerate(sorted(ESSENTIAL_TITLES.items()), 1):
        try:
            content = fetch_wikem_page(title)
        except Exception as e:
            failures.append((title, str(e)))
            print(f"  [{i}/{len(ESSENTIAL_TITLES)}] FAIL  {title}: {e}")
            continue

        if len(content) < MIN_CONTENT_CHARS:
            failures.append((title, f"content too short ({len(content)} chars)"))
            print(f"  [{i}/{len(ESSENTIAL_TITLES)}] thin  {title} ({len(content)} chars)")
            continue

        priority, tags = kings_day_tags_for(title)
        documents.append({
            "title": title,
            "category": cat_label,
            "content": content,
            "tags": tags,
            "priority": priority,
            "source": page_url(title),
        })
        flag = " [PRIORITY]" if priority == "high" else ""
        print(f"  [{i}/{len(ESSENTIAL_TITLES)}] ok    {title} ({len(content)} chars){flag}")
        time.sleep(SLEEP_BETWEEN_FETCHES)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pack_id": "core_medical",
        "version": "1.0",
        "documents": documents,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    high_priority = sum(1 for d in documents if d["priority"] == "high")
    print(f"\nWrote {OUTPUT}")
    print(f"  documents:           {len(documents)}")
    print(f"  king's day priority: {high_priority}")
    print(f"  failed/skipped:      {len(failures)}")
    if failures:
        print("\nFailures (review and adjust ESSENTIAL_TITLES if needed):")
        for title, reason in failures:
            print(f"  - {title}: {reason}")


if __name__ == "__main__":
    try:
        build()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
