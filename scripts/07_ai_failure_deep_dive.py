#!/usr/bin/env python3
"""AI-specific failure deep dive: Category B + C reports.

Subsets to algorithmic/processing and output/interpretation failures,
analyses sub-patterns, and extracts exemplar case narratives.

Outputs:
  - outputs/tables/ai_failure_subtypes.csv
  - outputs/tables/ai_failure_exemplars.csv
  - outputs/tables/ai_vs_generic_comparison.csv
  - outputs/tables/narrative_quality.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_classified():
    path = PROCESSED_DIR / "classified_reports.parquet"
    if path.exists():
        return pd.read_parquet(path)
    linked = pd.read_parquet(PROCESSED_DIR / "linked_reports.parquet")
    progress = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")
    return linked.merge(progress, on="report_number", how="inner")


def ai_specific_subset(classified):
    """Reports classified as AI-specific (B or C primary, or ai_specific=True)."""
    ai = classified[classified["ai_specific"] == True].copy()
    print(f"=== AI-Specific Reports ===")
    print(f"Total: {len(ai):,} / {len(classified):,} ({len(ai)/len(classified)*100:.1f}%)")
    print(f"\nPrimary category distribution (AI-specific only):")
    print(ai["primary_category"].value_counts().to_string())
    print(f"\nSpecialty distribution (AI-specific):")
    print(ai["matched_panel"].value_counts().head(10).to_string())
    return ai


def bc_analysis(classified):
    """Deep analysis of Category B (algorithmic) and C (output) failures."""
    has_b = classified["categories"].str.contains("B", na=False)
    has_c = classified["categories"].str.contains("C", na=False)
    bc = classified[has_b | has_c].copy()

    print(f"\n=== Category B + C Reports ===")
    print(f"Total B or C: {len(bc):,}")
    print(f"  B only: {(has_b & ~has_c).sum():,}")
    print(f"  C only: {(~has_b & has_c).sum():,}")
    print(f"  Both B+C: {(has_b & has_c).sum():,}")

    bc_with_harm = bc[bc["categories"].str.contains("G", na=False)]
    print(f"\n  B/C with patient harm (G): {len(bc_with_harm):,} ({len(bc_with_harm)/len(bc)*100:.1f}%)")

    print(f"\nSpecialty distribution (B/C reports):")
    print(bc["matched_panel"].value_counts().to_string())

    print(f"\nCo-occurring categories with B/C:")
    co_cats = bc["categories"].str.split(",").explode()
    co_cats = co_cats[~co_cats.isin(["B", "C"])]
    print(co_cats.value_counts().to_string())

    return bc


def narrative_quality_analysis(classified):
    """Assess MAUDE narrative quality for root-cause identification."""
    classified = classified.copy()
    classified["narrative_len"] = classified["narrative"].str.len()
    classified["has_narrative"] = classified["narrative"].notna() & (classified["narrative_len"] > 50)

    boilerplate_phrases = [
        "investigation pending",
        "if information is provided in the future",
        "a supplemental report will be issued",
        "product evaluation is pending",
        "no additional information",
        "investigation is ongoing",
    ]

    def is_boilerplate(text):
        if pd.isna(text):
            return True
        text_lower = text.lower()
        matches = sum(1 for p in boilerplate_phrases if p in text_lower)
        return matches >= 2 and len(text) < 500

    classified["is_boilerplate"] = classified["narrative"].apply(is_boilerplate)

    ai_keywords = [
        "algorithm", "artificial intelligence", "machine learning", "deep learning",
        "neural network", "classification", "detection", "false positive",
        "false negative", "sensitivity", "specificity", "ai ", "ml ",
        "software", "automated", "computer-aided", "cad",
    ]

    def mentions_ai(text):
        if pd.isna(text):
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in ai_keywords)

    classified["mentions_ai_terms"] = classified["narrative"].apply(mentions_ai)

    quality = pd.DataFrame([
        {"metric": "Reports with narrative >50 chars", "value": classified["has_narrative"].sum(),
         "pct": round(classified["has_narrative"].mean() * 100, 1)},
        {"metric": "Median narrative length (chars)", "value": int(classified["narrative_len"].median()),
         "pct": None},
        {"metric": "Boilerplate-only narratives", "value": classified["is_boilerplate"].sum(),
         "pct": round(classified["is_boilerplate"].mean() * 100, 1)},
        {"metric": "Mentions AI/ML terms", "value": classified["mentions_ai_terms"].sum(),
         "pct": round(classified["mentions_ai_terms"].mean() * 100, 1)},
        {"metric": "Classified as uninformative (U)", "value": (classified["primary_category"] == "U").sum(),
         "pct": round((classified["primary_category"] == "U").mean() * 100, 1)},
    ])

    quality.to_csv(TABLES_DIR / "narrative_quality.csv", index=False)
    print(f"\n=== Narrative Quality ===")
    print(quality.to_string(index=False))

    ai_specific = classified[classified["ai_specific"] == True]
    print(f"\nAmong AI-specific reports:")
    print(f"  Mentions AI terms: {ai_specific['mentions_ai_terms'].mean()*100:.1f}%")
    print(f"  Median narrative length: {ai_specific['narrative_len'].median():.0f} chars")

    return quality


def ai_vs_generic_comparison(classified):
    """Compare AI-specific vs generic device failures."""
    ai = classified[classified["ai_specific"] == True]
    generic = classified[classified["ai_specific"] == False]

    comparison = []
    for label, subset in [("AI-specific", ai), ("Generic device", generic)]:
        if len(subset) == 0:
            continue
        row = {
            "type": label,
            "n_reports": len(subset),
            "pct_total": round(len(subset) / len(classified) * 100, 1),
            "median_narrative_len": int(subset["narrative"].str.len().median()),
            "pct_with_harm_G": round(subset["categories"].str.contains("G", na=False).mean() * 100, 1),
        }
        for cat in ["A", "B", "C", "D", "E", "F", "G", "U"]:
            row[f"pct_primary_{cat}"] = round(
                (subset["primary_category"] == cat).mean() * 100, 1
            )
        comparison.append(row)

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(TABLES_DIR / "ai_vs_generic_comparison.csv", index=False)
    print(f"\n=== AI-Specific vs Generic Comparison ===")
    print(comp_df.T.to_string())
    return comp_df


def extract_exemplars(bc_reports, n=20):
    """Extract top exemplar narratives for manuscript case descriptions."""
    bc_ai = bc_reports[bc_reports["ai_specific"] == True].copy()
    bc_ai["narrative_len"] = bc_ai["narrative"].str.len()
    bc_ai = bc_ai[bc_ai["narrative_len"] > 200]
    bc_ai = bc_ai.sort_values("narrative_len", ascending=False)

    exemplars = []
    seen_devices = set()
    for _, row in bc_ai.iterrows():
        device = row.get("matched_device", "")
        if device in seen_devices:
            continue
        seen_devices.add(device)
        exemplars.append({
            "report_number": row["report_number"],
            "specialty": row.get("matched_panel", ""),
            "device": row.get("matched_device", ""),
            "company": row.get("matched_company", ""),
            "categories": row["categories"],
            "primary": row["primary_category"],
            "narrative_excerpt": str(row["narrative"])[:500],
        })
        if len(exemplars) >= n:
            break

    exemplar_df = pd.DataFrame(exemplars)
    exemplar_df.to_csv(TABLES_DIR / "ai_failure_exemplars.csv", index=False)
    print(f"\n=== Exemplar AI Failure Narratives ({len(exemplar_df)} reports) ===")
    for _, e in exemplar_df.head(5).iterrows():
        print(f"\n[{e['report_number']}] {e['device']} ({e['specialty']})")
        print(f"  Categories: {e['categories']} | Primary: {e['primary']}")
        print(f"  {e['narrative_excerpt'][:200]}...")
    return exemplar_df


def main():
    print("Loading classified data...")
    classified = load_classified()
    print(f"Loaded {len(classified):,} classified reports")

    ai = ai_specific_subset(classified)
    bc = bc_analysis(classified)
    narrative_quality_analysis(classified)
    ai_vs_generic_comparison(classified)
    extract_exemplars(bc)

    print(f"\nAll deep-dive tables saved to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
