#!/usr/bin/env python3
"""Link FDA AI/ML devices to MAUDE reports via product code + manufacturer disambiguation.

Linkage stages:
  1. Product code match (already done by 02_query_maude.py — each JSONL is one code)
  2. Manufacturer name fuzzy match: compare FDA device list "company" to MAUDE
     "manufacturer_d_name" using token-set ratio (handles word order, suffixes)
  3. For unmatched reports, attempt brand_name substring match against device names

Outputs:
  - linked_reports.parquet: MAUDE reports matched to specific AI devices
  - unlinked_reports.parquet: reports in matching product codes but no manufacturer match
  - linkage_summary.csv: per-device linkage statistics
"""

from pathlib import Path
import json
import pandas as pd
from thefuzz import fuzz
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MAUDE_DIR = PROJECT_ROOT / "data" / "raw" / "maude_by_code"

MFR_MATCH_THRESHOLD = 85
BRAND_CROSS_CHECK_THRESHOLD = 60


def load_maude_reports() -> pd.DataFrame:
    """Load all MAUDE JSONL files into a flat DataFrame."""
    rows = []
    for jsonl_path in sorted(MAUDE_DIR.glob("*.jsonl")):
        code = jsonl_path.stem
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                devices = rec.get("device", [])
                mdr_texts = rec.get("mdr_text", [])
                narrative = " ".join(
                    t.get("text", "") for t in mdr_texts
                    if t.get("text_type_code") in (
                        "Description of Event or Problem",
                        "Additional Manufacturer Narrative",
                        "Manufacturer Evaluation Summary",
                    )
                ).strip()

                for dev in devices:
                    rows.append({
                        "report_number": rec.get("report_number", ""),
                        "event_type": rec.get("event_type", ""),
                        "date_received": rec.get("date_received", ""),
                        "product_code_queried": code,
                        "device_report_product_code": dev.get("device_report_product_code", ""),
                        "brand_name": (dev.get("brand_name") or "").strip(),
                        "generic_name": (dev.get("generic_name") or "").strip(),
                        "manufacturer_d_name": (dev.get("manufacturer_d_name") or "").strip(),
                        "model_number": (dev.get("model_number") or "").strip(),
                        "catalog_number": (dev.get("catalog_number") or "").strip(),
                        "narrative": narrative,
                        "date_of_event": rec.get("date_of_event", ""),
                        "report_source_code": rec.get("report_source_code", ""),
                        "type_of_report": ",".join(rec.get("type_of_report", [])),
                        "product_problem_flag": rec.get("product_problem_flag", ""),
                        "event_location": rec.get("event_location", ""),
                    })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df):,} device-report rows from {len(list(MAUDE_DIR.glob('*.jsonl')))} JSONL files")
    return df


def load_device_list() -> pd.DataFrame:
    p = PROCESSED_DIR / "fda_aiml_devices.parquet"
    if not p.exists():
        p = PROCESSED_DIR / "fda_aiml_devices.csv"
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.copy()
    df.loc[:, "company_upper"] = df["company"].str.upper().str.strip()
    return df


def clean_name(name: str) -> str:
    """Normalize a company/manufacturer name for matching."""
    return (
        name.upper()
        .replace(",", "")
        .replace(".", "")
        .replace(" INC", "")
        .replace(" LLC", "")
        .replace(" LTD", "")
        .replace(" CORP", "")
        .replace(" CO", "")
        .replace("  ", " ")
        .strip()
    )


def _core_tokens(name: str) -> set[str]:
    """Extract meaningful tokens from a company name, dropping legal suffixes."""
    stop = {"INC", "LLC", "LTD", "CORP", "CO", "THE", "OF", "AND", "A", "AN"}
    return {t for t in clean_name(name).split() if t not in stop and len(t) > 1}


def link_reports(devices_df: pd.DataFrame, maude_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match MAUDE reports to AI devices by manufacturer name within product codes.

    Two-stage strategy:
      1. Manufacturer name match: token_set_ratio >= 85 between MAUDE
         manufacturer_d_name and FDA device list company.
      2. Brand-name cross-check (for borderline 75-84 manufacturer matches only):
         accept if brand_name also partially matches the device name.

    Reports where the best manufacturer score is <75 are classified as unlinked
    (belonging to non-AI devices that share the product code).
    """

    code_to_devices = {}
    for _, row in devices_df.iterrows():
        pc = row["product_code"]
        if pd.isna(pc):
            continue
        code_to_devices.setdefault(pc, []).append({
            "submission_number": row["submission_number"],
            "company": row["company"],
            "company_clean": clean_name(str(row["company"])),
            "company_tokens": _core_tokens(str(row["company"])),
            "device_name": str(row.get("device_name", "")),
            "panel": row.get("panel", ""),
            "clearance_date": row.get("clearance_date", ""),
        })

    linked_rows = []
    unlinked_rows = []

    for _, report in tqdm(maude_df.iterrows(), total=len(maude_df), desc="Linking"):
        pc = report["product_code_queried"]
        mfr_raw = report["manufacturer_d_name"]
        mfr_clean = clean_name(mfr_raw) if mfr_raw else ""
        mfr_tokens = _core_tokens(mfr_raw) if mfr_raw else set()
        brand = report["brand_name"].upper() if report["brand_name"] else ""

        candidates = code_to_devices.get(pc, [])
        if not candidates:
            unlinked_rows.append(report.to_dict() | {"match_type": "no_ai_devices_for_code"})
            continue

        # Score all candidates on manufacturer name
        scored = []
        for cand in candidates:
            score = fuzz.token_set_ratio(mfr_clean, cand["company_clean"])
            token_overlap = len(mfr_tokens & cand["company_tokens"])
            scored.append((score, token_overlap, cand))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        best_score, best_overlap, best_match = scored[0]

        # High-confidence manufacturer match
        if best_score >= MFR_MATCH_THRESHOLD:
            linked_rows.append(
                report.to_dict()
                | {
                    "matched_submission": best_match["submission_number"],
                    "matched_company": best_match["company"],
                    "matched_device": best_match["device_name"],
                    "matched_panel": best_match["panel"],
                    "matched_clearance_date": best_match["clearance_date"],
                    "match_type": "manufacturer",
                    "match_score": best_score,
                }
            )
            continue

        # Borderline manufacturer match (75-84): accept only if brand cross-checks
        if best_score >= 75 and best_overlap >= 1 and brand:
            dev_upper = best_match["device_name"].upper()
            brand_score = fuzz.token_set_ratio(brand, dev_upper)
            if brand_score >= BRAND_CROSS_CHECK_THRESHOLD:
                linked_rows.append(
                    report.to_dict()
                    | {
                        "matched_submission": best_match["submission_number"],
                        "matched_company": best_match["company"],
                        "matched_device": best_match["device_name"],
                        "matched_panel": best_match["panel"],
                        "matched_clearance_date": best_match["clearance_date"],
                        "match_type": "manufacturer+brand",
                        "match_score": best_score,
                    }
                )
                continue

        unlinked_rows.append(
            report.to_dict() | {
                "match_type": "no_match",
                "best_mfr_score": best_score,
            }
        )

    linked_df = pd.DataFrame(linked_rows)
    unlinked_df = pd.DataFrame(unlinked_rows)
    return linked_df, unlinked_df


def build_summary(devices_df: pd.DataFrame, linked_df: pd.DataFrame) -> pd.DataFrame:
    """Per-device linkage statistics."""
    if linked_df.empty:
        return pd.DataFrame()
    counts = linked_df.groupby("matched_submission").agg(
        n_reports=("report_number", "nunique"),
        n_narratives=("narrative", lambda x: (x.str.len() > 0).sum()),
        match_types=("match_type", lambda x: ",".join(sorted(x.unique()))),
    ).reset_index()

    summary = devices_df.merge(
        counts,
        left_on="submission_number",
        right_on="matched_submission",
        how="left",
    )
    summary = summary.copy()
    summary.loc[:, "n_reports"] = summary["n_reports"].fillna(0).astype(int)
    summary.loc[:, "n_narratives"] = summary["n_narratives"].fillna(0).astype(int)
    return summary.drop(columns=["matched_submission"], errors="ignore")


def main():
    devices_df = load_device_list()
    maude_df = load_maude_reports()

    if maude_df.empty:
        print("No MAUDE reports found. Run 02_query_maude.py first.")
        return

    print(f"\nLinking {len(maude_df):,} report-device rows to {len(devices_df):,} AI devices...")
    linked_df, unlinked_df = link_reports(devices_df, maude_df)

    linked_path = PROCESSED_DIR / "linked_reports.parquet"
    unlinked_path = PROCESSED_DIR / "unlinked_reports.parquet"
    summary_path = PROCESSED_DIR / "linkage_summary.csv"

    if not linked_df.empty:
        linked_df.to_parquet(linked_path, index=False)
    if not unlinked_df.empty:
        unlinked_df.to_parquet(unlinked_path, index=False)

    summary = build_summary(devices_df, linked_df)
    summary.to_csv(summary_path, index=False)

    n_linked_reports = linked_df["report_number"].nunique() if not linked_df.empty else 0
    n_unlinked = len(unlinked_df)
    n_devices_with = (summary["n_reports"] > 0).sum() if not summary.empty else 0
    n_devices_zero = (summary["n_reports"] == 0).sum() if not summary.empty else 0

    print(f"\n{'='*60}")
    print(f"Linked reports (unique):   {n_linked_reports:,}")
    print(f"Unlinked reports:          {n_unlinked:,}")
    print(f"AI devices with reports:   {n_devices_with:,} / {len(devices_df):,}")
    print(f"AI devices with ZERO:      {n_devices_zero:,} / {len(devices_df):,} ({100*n_devices_zero/len(devices_df):.1f}%)")
    print(f"{'='*60}")
    print(f"Outputs:")
    print(f"  {linked_path.relative_to(PROJECT_ROOT)}")
    print(f"  {unlinked_path.relative_to(PROJECT_ROOT)}")
    print(f"  {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
