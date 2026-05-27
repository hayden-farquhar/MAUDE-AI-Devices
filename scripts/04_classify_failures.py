#!/usr/bin/env python3
"""Classify linked MAUDE reports into AI-specific failure mode taxonomy.

Uses GPT-4o-mini with few-shot prompting for multi-label classification
across 7 failure categories. Processes in batches with checkpointing.

Outputs:
  - data/processed/classified_reports.parquet
  - data/processed/classification_progress.csv (for resumability)
"""

from pathlib import Path
import asyncio
import json
import time
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

load_dotenv(PROJECT_ROOT / ".env")

aclient = AsyncOpenAI()
MODEL = "gpt-4o-mini"
CONCURRENCY = 20
CHECKPOINT_EVERY = 500
MAX_NARRATIVE_CHARS = 3000

SYSTEM_PROMPT = """You are an expert medical device safety analyst classifying FDA MAUDE adverse event reports for AI/ML-enabled medical devices.

For each report narrative, assign ALL applicable failure mode categories from this taxonomy:

A — DATA INPUT FAILURE: Problems with data entering the device. Includes: poor image/signal quality, missing or corrupt input data, sensor malfunction, wrong modality or patient positioning, data acquisition error.

B — ALGORITHMIC/PROCESSING FAILURE: The device's software or AI/ML algorithm produced an incorrect or unexpected result. Includes: false positive, false negative, misclassification, incorrect measurement/quantification, algorithm crash or hang, software bug in processing logic, concept drift, out-of-distribution failure. ALSO includes: firmware issues, power-on resets (POR), electrical resets, and any internal software/firmware malfunction — these are software failures (B), NOT infrastructure failures (E).

C — OUTPUT/INTERPRETATION FAILURE: The device produced output that was misleading, incomplete, or difficult to interpret correctly. ONLY use C when the output/display itself is the problem — not when the underlying algorithm is wrong (that's B). Includes: wrong patient/exam metadata attached to otherwise correct results, display rendering error, report failed to generate/upload. If the algorithm produced a wrong answer AND that answer was displayed, the primary issue is B, not C.

D — USER INTERACTION FAILURE: Problems arising from how the user interacted with or responded to the device. Includes: user interface confusion, automation bias (over-reliance on AI output), alert fatigue, training/competency gap, workaround or off-label use.

E — INFRASTRUCTURE/INTEGRATION FAILURE: Problems with the device's EXTERNAL technical environment, NOT the device's own software. Includes: network/connectivity loss, EHR/PACS integration error, DICOM compatibility issue, interoperability failure between separate systems. NOTE: Internal firmware failures, power-on resets, and software bugs within the device itself are B (algorithmic/processing), not E.

F — HARDWARE FAILURE: Physical device or component malfunction unrelated to the AI/ML algorithm. Includes: display failure, power supply issue, mechanical breakage, sensor/probe physical damage, overheating, battery failure.

G — PATIENT HARM: Any report of actual patient harm, regardless of cause. Includes: death, serious injury requiring intervention, injury not requiring intervention, prolonged procedure, delayed diagnosis/treatment, unnecessary procedure performed.

U — UNINFORMATIVE: The narrative contains ABSOLUTELY NO information about what went wrong. ONLY assign U when the entire narrative is regulatory boilerplate with zero description of a device event. If there is even ONE sentence describing what happened (e.g., "firmware issue observed", "electrical reset occurred", "signal quality change", "artifact on episode"), classify based on that sentence — do NOT assign U. Read through all boilerplate text carefully to find any event description before assigning U.

Rules:
- Assign ALL categories that apply (multi-label). Most reports will have 1-3 categories.
- Category G (patient harm) should ALWAYS be co-assigned with the causal category when identifiable. G should NEVER be the primary category — the primary should be the CAUSE of harm (A, B, C, D, E, or F). For example, if a patient was burned by an MRI (hardware issue causing harm), the categories are F,G with primary F.
- Category U should only be assigned alone — if you can extract ANY failure mode from the narrative, do not assign U. Read carefully through boilerplate text; many Medtronic reports bury the event description after several paragraphs of regulatory language.
- Base classification strictly on what the narrative describes, not on speculation.
- Firmware failures, power-on resets, and electrical resets are B (software/algorithm), not E (infrastructure).

Respond with ONLY a JSON object: {"categories": ["A", "B", ...], "primary": "X", "confidence": "high|medium|low", "ai_specific": true/false}

- "primary": the single most important failure category for this report. NEVER assign G as primary — use the causal category instead.
- "confidence": your confidence in the classification
- "ai_specific": true if the failure is specifically related to AI/ML functionality (not just a generic device failure that happens to be on an AI-enabled device)"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": "Narrative: THE CUSTOMER REPORTED THAT PATIENT WAS POSITIONED SUPINE HEADFIRST, BUT THE SYSTEM RECOGNIZED THE POSITION INCORRECTLY TO PRONE HEADFIRST. ISSUE WAS IDENTIFIED BY THE TECHNICIAN BEFORE THE ACQUISITION AND HAD THE POSITION MANUALLY CORRECTED AFTERWARDS. THERE WAS NO PATIENT INJURY OR HARM REPORTED. ENGINEERING INVESTIGATION CONCLUDED THAT THE CAUSE IS A SOFTWARE ISSUE WHERE THE ALGORITHM MAY INCORRECTLY DETERMINE THE PATIENT ORIENTATION."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["B", "C"], "primary": "B", "confidence": "high", "ai_specific": true}'
    },
    {
        "role": "user",
        "content": "Narrative: THE PATIENT EXPERIENCED AN ARRHYTHMIA THAT MET MEDICAL DOCTOR NOTIFICATION REQUIREMENTS THAT WAS NOT COMMUNICATED DURING THE WEAR PERIOD. THE INVESTIGATION REVEALED THAT A PRELIMINARY ECG INTERPRETATION PROVIDED TO THE PHYSICIAN WAS MISCLASSIFIED. THE HEALTHCARE PROVIDER WAS IMMEDIATELY NOTIFIED, AND THE PATIENT HAD RECEIVED A PACEMAKER DUE TO THE INITIAL INTERPRETATION BEING INCORRECT."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["B", "C", "G"], "primary": "B", "confidence": "high", "ai_specific": true}'
    },
    {
        "role": "user",
        "content": "Narrative: THE AC INLET SOLDER JOINT HAS BEEN DAMAGED BY EXCESSIVE MECHANICAL STRESS EXERTED ON THE JOINT VIA THE AC CORD. CUSTOMER REPORTED THAT THEY SAW SMOKE COMING OUT OF THE DC ADAPTER. THEY IMMEDIATELY UNPLUGGED THE CORD AND DID NOT USE IT AGAIN. THERE WERE NO INJURIES OR OTHER DAMAGE REPORTED."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["F"], "primary": "F", "confidence": "high", "ai_specific": false}'
    },
    {
        "role": "user",
        "content": "Narrative: PRODUCT EVENT SUMMARY: THE DEVICE WAS RETURNED AND ANALYZED. THERE WAS AN ISSUE OBSERVED WITH THE FIRMWARE. MEDTRONIC SUBMITS THIS REPORT TO COMPLY WITH FDA REGULATIONS 21 CFR PARTS 4 AND 803. MEDTRONIC HAS MADE REASONABLE EFFORTS TO PROVIDE AS MUCH RELEVANT INFORMATION AS IS AVAILABLE TO THE COMPANY AS OF THE SUBMISSION DATE OF THIS REPORT."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["B"], "primary": "B", "confidence": "medium", "ai_specific": false}'
    },
    {
        "role": "user",
        "content": "Narrative: PHILIPS RECEIVED A REPORT ON A HEATING INCIDENT ON AN INGENIA 1.5T R SYSTEM. A MALE PATIENT WAS POSITIONED FEET FIRST SUPINE TO UNDERGO A PELVIS EXAMINATION WITH THE ANTERIOR AND POSTERIOR COIL. MORE THAN 1 HOUR AFTER THE EXAMINATION BLISTERING WAS OBSERVED IN THE GROIN AREA OF THE PATIENT."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["F", "G"], "primary": "F", "confidence": "high", "ai_specific": false}'
    },
    {
        "role": "user",
        "content": "Narrative: PRODUCT EVENT SUMMARY: THE DEVICE WAS NOT RETURNED FOR ANALYSIS, HOWEVER, PERFORMANCE DATA COLLECTED FROM THE DEVICE WAS RECEIVED AND ANALYZED. ANALYSIS OF THE DEVICE MEMORY INDICATED A PARTIAL ELECTRICAL RESET OCCURRED. MEDTRONIC SUBMITS THIS REPORT TO COMPLY WITH FDA REGULATIONS 21 CFR PARTS 4 AND 803."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["B"], "primary": "B", "confidence": "medium", "ai_specific": false}'
    },
    {
        "role": "user",
        "content": "Narrative: ECHONOUS DETECTED A SOFTWARE ANOMALY IN THE MODULE RESPONSIBLE FOR CREATING DICOM FILES FOR EXPORT OR ARCHIVE. ALTHOUGH THE IMAGES AND CLIPS ARE BEING STORED CORRECTLY IN THE DICOM FILES, THE PATIENT DEMOGRAPHIC AND STUDY INFORMATION ATTACHED TO THE DICOM FILE MAY NOT ALWAYS MATCH THE CORRECT PATIENT."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["C", "E"], "primary": "E", "confidence": "high", "ai_specific": false}'
    },
    {
        "role": "user",
        "content": "Narrative: A RISK TO THE PATIENT COULD NOT BE EXCLUDED SINCE SCREWS WERE PLACED IN A DIFFERENT POSITION THAN DESIRED WITH THE NAVIGATION DEVICE INVOLVED. THE SURGEON REPORTED THE TWO SCREWS PLACED NOT AS INTENDED WERE REVISED IN THE SAME SURGERY. ALL SCREWS WERE IN CORRECT FINAL POSITIONS. THE REGISTRATION ACCURACY METRIC WAS WITHIN SPECIFICATION AT THE START BUT DEGRADED DURING THE PROCEDURE."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["B", "C", "G"], "primary": "B", "confidence": "high", "ai_specific": true}'
    },
    {
        "role": "user",
        "content": "Narrative: IF INFORMATION IS PROVIDED IN THE FUTURE, A SUPPLEMENTAL REPORT WILL BE ISSUED. THE DEVICE HAS BEEN RETURNED AND THE PRODUCT EVALUATION IS PENDING."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["U"], "primary": "U", "confidence": "high", "ai_specific": false}'
    },
    {
        "role": "user",
        "content": "Narrative: USER REPORTED THAT DURING A CARDIAC ARREST EMERGENCY USE, WHEN ATTEMPTING TO APPLY PADS, THE DEVICE REPEATEDLY INDICATED THAT THE PATIENT WAS NOT DETECTED. USER STATED THE PATIENT WAS DRY, WITH NO EXCESSIVE BODY HAIR OR OTHER OBVIOUS IMPEDIMENTS TO PAD PLACEMENT."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["A", "B"], "primary": "A", "confidence": "medium", "ai_specific": true}'
    },
    {
        "role": "user",
        "content": "Narrative: THE VISUALASE SYSTEM DOES NOT RECOGNIZE OR CANNOT PROCESS ONE OF THE PLANES IN THE 2 PLANE TMAP. THE VISUALASE SYSTEM BEHAVES NORMALLY FOR ONE PLANE, DOES NOT DISPLAY THE OTHER PLANE, AND EVENTUALLY TIMES OUT. THIS PREVENTS THE ABLATION PROCESS FROM TAKING PLACE. ONE RESOLUTION THAT WORKS SOMETIMES IS TO ROTATE THE PROBLEMATIC TMAP PLANE SLIGHTLY AND RERUN."
    },
    {
        "role": "assistant",
        "content": '{"categories": ["E", "B"], "primary": "E", "confidence": "medium", "ai_specific": false}'
    },
]


async def classify_narrative(narrative: str, semaphore: asyncio.Semaphore) -> dict:
    """Classify a single narrative using GPT-4o-mini (async)."""
    truncated = narrative[:MAX_NARRATIVE_CHARS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user", "content": f"Narrative: {truncated}"},
    ]

    async with semaphore:
        try:
            response = await aclient.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            result["_tokens"] = response.usage.total_tokens if response.usage else 0
            return result
        except Exception as e:
            return {"categories": ["U"], "primary": "U", "confidence": "low", "ai_specific": False, "_error": str(e)}


async def classify_batch(rows: list[dict], semaphore: asyncio.Semaphore, pbar: tqdm) -> list[dict]:
    """Classify a batch of rows concurrently."""
    async def process_one(row):
        narrative = row["narrative"]
        if not narrative or len(str(narrative)) < 20:
            result = {"categories": ["U"], "primary": "U", "confidence": "high", "ai_specific": False}
        else:
            result = await classify_narrative(str(narrative), semaphore)

        tokens = result.pop("_tokens", 0)
        error = result.pop("_error", None)
        pbar.update(1)
        return {
            "report_number": row["report_number"],
            "categories": ",".join(result.get("categories", ["U"])),
            "primary_category": result.get("primary", "U"),
            "confidence": result.get("confidence", "low"),
            "ai_specific": result.get("ai_specific", False),
            "_tokens": tokens,
            "_error": error,
        }

    tasks = [process_one(row) for row in rows]
    return await asyncio.gather(*tasks)


async def async_main():
    linked = pd.read_parquet(PROCESSED_DIR / "linked_reports.parquet")
    progress_path = PROCESSED_DIR / "classification_progress.csv"

    already_done = set()
    if progress_path.exists():
        done_df = pd.read_csv(progress_path)
        already_done = set(done_df["report_number"].unique())
        print(f"Resuming: {len(already_done):,} reports already classified")

    to_classify = linked[~linked["report_number"].isin(already_done)].copy()
    to_classify = to_classify.drop_duplicates(subset="report_number")
    print(f"Reports to classify: {len(to_classify):,}")

    if to_classify.empty:
        print("All reports already classified.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    rows = to_classify.to_dict("records")
    total_tokens = 0
    errors = 0
    all_results = []

    pbar = tqdm(total=len(rows), desc="Classifying")

    for chunk_start in range(0, len(rows), CHECKPOINT_EVERY):
        chunk = rows[chunk_start : chunk_start + CHECKPOINT_EVERY]
        batch_results = await classify_batch(chunk, semaphore, pbar)

        for r in batch_results:
            tokens = r.pop("_tokens", 0)
            error = r.pop("_error", None)
            total_tokens += tokens
            if error:
                errors += 1
            all_results.append(r)

        # Checkpoint
        batch_df = pd.DataFrame(all_results)
        if progress_path.exists():
            existing = pd.read_csv(progress_path)
            batch_df = pd.concat([existing, batch_df], ignore_index=True)
        batch_df.to_csv(progress_path, index=False)
        all_results = []

        cost_est = total_tokens / 1_000_000 * 0.15
        pbar.set_postfix(cost=f"${cost_est:.2f}", errors=errors)

    pbar.close()

    # Merge with linked data and save final output
    all_classifications = pd.read_csv(progress_path)
    merged = linked.merge(all_classifications, on="report_number", how="left")
    merged.to_parquet(PROCESSED_DIR / "classified_reports.parquet", index=False)

    cost_est = total_tokens / 1_000_000 * 0.15
    print(f"\nDone. {len(all_classifications):,} reports classified.")
    print(f"Tokens used: {total_tokens:,} (~${cost_est:.2f})")
    print(f"Errors: {errors}")

    print("\nCategory distribution:")
    all_cats = all_classifications["categories"].str.split(",").explode()
    print(all_cats.value_counts().to_string())
    print(f"\nAI-specific reports: {all_classifications['ai_specific'].sum():,} / {len(all_classifications):,}")
    print(f"\nPrimary category distribution:")
    print(all_classifications["primary_category"].value_counts().to_string())


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
