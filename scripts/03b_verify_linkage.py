#!/usr/bin/env python3
"""Manual verification of linkage on a stratified 50-device sample.

Checks:
  A. Devices WITH linked reports (n=15): are the linked reports genuinely from
     the AI device manufacturer? Cross-checks manufacturer_d_name vs company.
  B. Zero-report devices whose product code HAS unlinked reports (n=20):
     are there unlinked reports that should have matched? (false-negative check)
  C. True-zero devices — product code returned no MAUDE reports (n=15):
     confirms these product codes genuinely have no MAUDE data.

Outputs:
  - data/processed/verification_sample.csv: full verification table
  - prints summary verdicts to stdout
"""

from pathlib import Path
import pandas as pd
from thefuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def clean_name(name: str) -> str:
    return (
        name.upper()
        .replace(",", "").replace(".", "")
        .replace(" INC", "").replace(" LLC", "").replace(" LTD", "")
        .replace(" CORP", "").replace(" CO", "").replace("  ", " ")
        .strip()
    )


def main():
    linked = pd.read_parquet(PROCESSED_DIR / "linked_reports.parquet")
    unlinked = pd.read_parquet(PROCESSED_DIR / "unlinked_reports.parquet")
    summary = pd.read_csv(PROCESSED_DIR / "linkage_summary.csv")

    # ── Stratified sampling ──────────────────────────────────────────
    with_reports = summary[summary["n_reports"] > 0].copy()
    zero_with_unlinked = summary[
        (summary["n_reports"] == 0)
        & summary["product_code"].isin(unlinked["product_code_queried"].unique())
    ].copy()
    true_zero = summary[
        (summary["n_reports"] == 0)
        & ~summary["product_code"].isin(unlinked["product_code_queried"].unique())
    ].copy()

    # Stratum A: 15 devices with reports (5 high ≥100, 5 mid 10-99, 5 low 1-9)
    high = with_reports[with_reports["n_reports"] >= 100].sample(min(5, len(with_reports[with_reports["n_reports"] >= 100])), random_state=1)
    mid = with_reports[(with_reports["n_reports"] >= 10) & (with_reports["n_reports"] < 100)].sample(min(5, len(with_reports[(with_reports["n_reports"] >= 10) & (with_reports["n_reports"] < 100)])), random_state=2)
    low = with_reports[with_reports["n_reports"] < 10].sample(min(5, len(with_reports[with_reports["n_reports"] < 10])), random_state=3)
    stratum_a = pd.concat([high, mid, low])

    # Stratum B: 20 zero-report devices with unlinked reports in their code
    stratum_b = zero_with_unlinked.sample(min(20, len(zero_with_unlinked)), random_state=4)

    # Stratum C: 15 true-zero devices
    stratum_c = true_zero.sample(min(15, len(true_zero)), random_state=5)

    results = []

    # ── Stratum A: verify linked reports ─────────────────────────────
    print("=" * 80)
    print("STRATUM A: Devices WITH linked reports (verify true positives)")
    print("=" * 80)

    for _, dev in stratum_a.iterrows():
        sub = dev["submission_number"]
        dev_reports = linked[linked["matched_submission"] == sub]
        n = len(dev_reports)
        company = str(dev["company"])
        company_clean = clean_name(company)

        mfr_names = dev_reports["manufacturer_d_name"].value_counts().head(3)
        best_mfr = mfr_names.index[0] if len(mfr_names) > 0 else ""
        best_mfr_score = fuzz.token_set_ratio(clean_name(best_mfr), company_clean)

        match_types = dev_reports["match_type"].value_counts().to_dict()
        scores = dev_reports["match_score"]

        # Verdict
        if best_mfr_score >= 85:
            verdict = "CONFIRMED"
        elif best_mfr_score >= 70:
            verdict = "PLAUSIBLE"
        else:
            verdict = "SUSPECT"

        print(f"\n  {sub} | {str(dev['device_name'])[:45]} | {n} reports")
        print(f"    FDA company:  {company[:50]}")
        print(f"    MAUDE mfr(s): {'; '.join(f'{m[:40]}({c})' for m, c in mfr_names.items())}")
        print(f"    Cross-check:  {best_mfr_score} → {verdict}")
        print(f"    Match types:  {match_types} | Score range: {scores.min():.0f}-{scores.max():.0f}")

        sample_narr = ""
        with_narr = dev_reports[dev_reports["narrative"].str.len() > 20]
        if not with_narr.empty:
            sample_narr = with_narr.iloc[0]["narrative"][:200]

        results.append({
            "stratum": "A",
            "submission_number": sub,
            "device_name": dev["device_name"],
            "company": company,
            "panel": dev["panel"],
            "n_linked_reports": n,
            "top_maude_mfr": best_mfr,
            "mfr_cross_check_score": best_mfr_score,
            "match_types": str(match_types),
            "verdict": verdict,
            "sample_narrative": sample_narr,
        })

    # ── Stratum B: verify zero-report devices (false-negative check) ─
    print("\n" + "=" * 80)
    print("STRATUM B: Zero-report devices with unlinked reports in product code")
    print("(checking for missed matches — false negatives)")
    print("=" * 80)

    for _, dev in stratum_b.iterrows():
        sub = dev["submission_number"]
        pc = dev["product_code"]
        company = str(dev["company"])
        company_clean = clean_name(company)

        code_unlinked = unlinked[unlinked["product_code_queried"] == pc]
        n_unlinked = len(code_unlinked)

        # Check the top manufacturer names in unlinked reports for this code
        top_mfrs = code_unlinked["manufacturer_d_name"].value_counts().head(5)
        best_unlinked_score = 0
        best_unlinked_mfr = ""
        for mfr_name in top_mfrs.index:
            score = fuzz.token_set_ratio(clean_name(str(mfr_name)), company_clean)
            if score > best_unlinked_score:
                best_unlinked_score = score
                best_unlinked_mfr = mfr_name

        if best_unlinked_score >= 85:
            verdict = "FALSE NEGATIVE — missed match"
        elif best_unlinked_score >= 70:
            verdict = "BORDERLINE — review needed"
        else:
            verdict = "CONFIRMED ZERO — no matching mfr"

        print(f"\n  {sub} | {str(dev['device_name'])[:45]} | code {pc} ({n_unlinked} unlinked)")
        print(f"    AI device company: {company[:50]}")
        print(f"    Top unlinked mfrs: {'; '.join(f'{m[:35]}({c})' for m, c in top_mfrs.head(3).items())}")
        print(f"    Best unlinked match: \"{best_unlinked_mfr[:40]}\" score={best_unlinked_score} → {verdict}")

        results.append({
            "stratum": "B",
            "submission_number": sub,
            "device_name": dev["device_name"],
            "company": company,
            "panel": dev["panel"],
            "product_code": pc,
            "n_unlinked_in_code": n_unlinked,
            "best_unlinked_mfr": best_unlinked_mfr,
            "best_unlinked_score": best_unlinked_score,
            "verdict": verdict,
        })

    # ── Stratum C: true-zero confirmation ────────────────────────────
    print("\n" + "=" * 80)
    print("STRATUM C: True-zero devices (product code has no MAUDE reports)")
    print("=" * 80)

    from pathlib import Path as P
    maude_dir = PROJECT_ROOT / "data" / "raw" / "maude_by_code"

    for _, dev in stratum_c.iterrows():
        sub = dev["submission_number"]
        pc = dev["product_code"]
        jsonl = maude_dir / f"{pc}.jsonl"
        file_lines = 0
        if jsonl.exists():
            file_lines = sum(1 for _ in open(jsonl))

        verdict = "CONFIRMED ZERO" if file_lines == 0 else f"UNEXPECTED: {file_lines} records in JSONL"

        print(f"  {sub} | {str(dev['device_name'])[:45]} | code {pc} | JSONL lines: {file_lines} → {verdict}")

        results.append({
            "stratum": "C",
            "submission_number": sub,
            "device_name": dev["device_name"],
            "company": dev["company"],
            "panel": dev["panel"],
            "product_code": pc,
            "jsonl_lines": file_lines,
            "verdict": verdict,
        })

    # ── Summary ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = PROCESSED_DIR / "verification_sample.csv"
    results_df.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    for stratum in ["A", "B", "C"]:
        sdf = results_df[results_df["stratum"] == stratum]
        print(f"\nStratum {stratum} (n={len(sdf)}):")
        print(sdf["verdict"].value_counts().to_string())

    print(f"\nSaved → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
