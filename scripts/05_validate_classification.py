#!/usr/bin/env python3
"""Validate LLM classification against manual review on 200-report subsample.

Outputs:
  - data/processed/validation_sample.csv (200 reports for manual labelling)
  - outputs/tables/validation_metrics.csv (per-category F1, Cohen's kappa)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "U"]


def create_validation_sample(n: int = 200, seed: int = 42):
    """Stratified sample of 200 reports for manual validation.

    Stratifies by primary_category so rare categories are represented.
    """
    classified = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")
    linked = pd.read_parquet(PROCESSED_DIR / "linked_reports.parquet")

    merged = classified.merge(
        linked[["report_number", "narrative", "matched_panel", "matched_device", "matched_company"]],
        on="report_number",
        how="left",
    )

    samples = []
    per_cat = max(n // len(CATEGORIES), 10)
    for cat in CATEGORIES:
        pool = merged[merged["primary_category"] == cat]
        k = min(per_cat, len(pool))
        if k > 0:
            samples.append(pool.sample(n=k, random_state=seed))

    sample = pd.concat(samples).drop_duplicates(subset="report_number")

    remaining = n - len(sample)
    if remaining > 0:
        extra_pool = merged[~merged["report_number"].isin(sample["report_number"])]
        samples.append(extra_pool.sample(n=min(remaining, len(extra_pool)), random_state=seed + 1))
        sample = pd.concat(samples).drop_duplicates(subset="report_number")

    sample = sample.head(n)

    sample["manual_categories"] = ""
    sample["manual_primary"] = ""
    sample["manual_ai_specific"] = ""
    sample["notes"] = ""

    out_path = PROCESSED_DIR / "validation_sample.csv"
    sample.to_csv(out_path, index=False)
    print(f"Validation sample saved: {out_path} ({len(sample)} reports)")
    print(f"Category distribution in sample:")
    print(sample["primary_category"].value_counts().to_string())
    return sample


def compute_validation_metrics():
    """Compute per-category F1 and Cohen's kappa from completed manual review."""
    sample = pd.read_csv(PROCESSED_DIR / "validation_sample.csv")

    if sample["manual_categories"].isna().all() or (sample["manual_categories"] == "").all():
        print("Manual labels not yet filled in. Run create_validation_sample first,")
        print("then fill manual_categories and manual_primary columns.")
        return None

    labelled = sample[sample["manual_categories"].notna() & (sample["manual_categories"] != "")]

    def parse_cats(s):
        if pd.isna(s) or s == "":
            return []
        return [c.strip() for c in str(s).split(",")]

    llm_cats = labelled["categories"].apply(parse_cats)
    manual_cats = labelled["manual_categories"].apply(parse_cats)

    mlb = MultiLabelBinarizer(classes=CATEGORIES)
    y_llm = mlb.fit_transform(llm_cats)
    y_manual = mlb.transform(manual_cats)

    results = []
    for i, cat in enumerate(CATEGORIES):
        tp = ((y_llm[:, i] == 1) & (y_manual[:, i] == 1)).sum()
        fp = ((y_llm[:, i] == 1) & (y_manual[:, i] == 0)).sum()
        fn = ((y_llm[:, i] == 0) & (y_manual[:, i] == 1)).sum()
        tn = ((y_llm[:, i] == 0) & (y_manual[:, i] == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = (y_manual[:, i] == 1).sum()
        results.append({
            "category": cat,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": int(support),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    results_df = pd.DataFrame(results)

    kappa_primary = cohen_kappa_score(
        labelled["primary_category"], labelled["manual_primary"]
    )

    y_llm_flat = y_llm.flatten()
    y_manual_flat = y_manual.flatten()
    kappa_multilabel = cohen_kappa_score(y_llm_flat, y_manual_flat)

    ai_agree = (labelled["ai_specific"].astype(bool) == labelled["manual_ai_specific"].map(
        lambda x: str(x).lower() in ("true", "1", "yes")
    )).mean()

    print(f"\n=== Validation Results ({len(labelled)} reports) ===")
    print(results_df.to_string(index=False))
    print(f"\nCohen's κ (primary category): {kappa_primary:.3f}")
    print(f"Cohen's κ (multi-label, flattened): {kappa_multilabel:.3f}")
    print(f"AI-specific agreement: {ai_agree:.1%}")

    macro_f1 = results_df["f1"].mean()
    weighted_f1 = (results_df["f1"] * results_df["support"]).sum() / results_df["support"].sum()
    print(f"\nMacro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")

    summary = pd.concat([
        results_df,
        pd.DataFrame([{
            "category": "OVERALL",
            "precision": results_df["precision"].mean(),
            "recall": results_df["recall"].mean(),
            "f1": macro_f1,
            "support": results_df["support"].sum(),
        }]),
    ])
    summary.to_csv(TABLES_DIR / "validation_metrics.csv", index=False)
    print(f"\nMetrics saved to {TABLES_DIR / 'validation_metrics.csv'}")

    return results_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "metrics":
        compute_validation_metrics()
    else:
        create_validation_sample()
