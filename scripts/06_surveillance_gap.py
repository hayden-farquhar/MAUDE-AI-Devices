#!/usr/bin/env python3
"""Surveillance adequacy analysis for FDA AI/ML devices in MAUDE.

Quantifies the surveillance gap: devices with zero reports, per-device
report distribution, temporal trends, and specialty-stratified analysis.

Outputs:
  - outputs/tables/surveillance_summary.csv
  - outputs/tables/device_report_counts.csv
  - outputs/tables/specialty_surveillance.csv
  - outputs/tables/top20_devices.csv
  - outputs/tables/temporal_trends.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "U"]


def load_data():
    devices = pd.read_csv(PROCESSED_DIR / "fda_aiml_devices.csv").copy()
    devices["clearance_date"] = pd.to_datetime(devices["clearance_date"])
    devices["clearance_year"] = devices["clearance_date"].dt.year

    linked = pd.read_parquet(PROCESSED_DIR / "linked_reports.parquet").copy()
    linked["date_received"] = pd.to_datetime(linked["date_received"], errors="coerce")
    linked["report_year"] = linked["date_received"].dt.year

    classified_path = PROCESSED_DIR / "classified_reports.parquet"
    if classified_path.exists():
        classified = pd.read_parquet(classified_path)
    else:
        progress = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")
        classified = linked.merge(progress, on="report_number", how="left")

    return devices, linked, classified


def device_report_counts(devices, linked):
    """Per-device report count distribution."""
    per_device = (
        linked.groupby("matched_submission")
        .agg(report_count=("report_number", "nunique"))
        .reset_index()
    )

    device_counts = devices.merge(
        per_device,
        left_on="submission_number",
        right_on="matched_submission",
        how="left",
    )
    device_counts["report_count"] = device_counts["report_count"].fillna(0).astype(int)

    device_counts.to_csv(TABLES_DIR / "device_report_counts.csv", index=False)

    n_total = len(device_counts)
    n_zero = (device_counts["report_count"] == 0).sum()
    n_1_10 = ((device_counts["report_count"] >= 1) & (device_counts["report_count"] <= 10)).sum()
    n_gt10 = (device_counts["report_count"] > 10).sum()
    n_gt100 = (device_counts["report_count"] > 100).sum()

    summary = pd.DataFrame([
        {"metric": "Total AI/ML devices", "value": n_total},
        {"metric": "Zero MAUDE reports", "value": n_zero},
        {"metric": "Zero reports (%)", "value": round(n_zero / n_total * 100, 1)},
        {"metric": "1-10 reports", "value": n_1_10},
        {"metric": ">10 reports", "value": n_gt10},
        {"metric": ">100 reports", "value": n_gt100},
        {"metric": "Median reports per device", "value": device_counts["report_count"].median()},
        {"metric": "Mean reports per device", "value": round(device_counts["report_count"].mean(), 1)},
        {"metric": "Max reports per device", "value": device_counts["report_count"].max()},
        {"metric": "Total linked reports", "value": device_counts["report_count"].sum()},
    ])

    print("=== Device-Level Surveillance Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(TABLES_DIR / "surveillance_summary.csv", index=False)

    return device_counts


def specialty_analysis(device_counts, classified):
    """Surveillance rates and failure modes by medical specialty."""
    specialty = (
        device_counts.groupby("panel")
        .agg(
            n_devices=("submission_number", "count"),
            n_with_reports=("report_count", lambda x: (x > 0).sum()),
            total_reports=("report_count", "sum"),
            median_reports=("report_count", "median"),
        )
        .reset_index()
    )
    specialty["zero_report_pct"] = round(
        (1 - specialty["n_with_reports"] / specialty["n_devices"]) * 100, 1
    )
    specialty = specialty.sort_values("n_devices", ascending=False)

    if "categories" in classified.columns:
        def get_primary_mode(group):
            cats = group["primary_category"].value_counts()
            return cats.index[0] if len(cats) > 0 else "N/A"

        cat_by_panel = (
            classified.groupby("matched_panel")
            .apply(get_primary_mode, include_groups=False)
            .reset_index()
        )
        cat_by_panel.columns = ["panel", "dominant_failure_mode"]

        ai_rate = (
            classified.groupby("matched_panel")["ai_specific"]
            .mean()
            .reset_index()
        )
        ai_rate.columns = ["panel", "ai_specific_rate"]
        ai_rate = ai_rate.copy()
        ai_rate["ai_specific_rate"] = (ai_rate["ai_specific_rate"] * 100).round(1)

        specialty = specialty.merge(cat_by_panel, on="panel", how="left")
        specialty = specialty.merge(ai_rate, on="panel", how="left")

    specialty.to_csv(TABLES_DIR / "specialty_surveillance.csv", index=False)
    print("\n=== Specialty-Level Surveillance ===")
    print(specialty.to_string(index=False))
    return specialty


def top_20_devices(device_counts, classified):
    """Top 20 devices by MAUDE report count with failure mode breakdown."""
    top20 = device_counts.nlargest(20, "report_count")[
        ["submission_number", "device_name", "company_clean", "panel", "product_code", "clearance_date", "report_count"]
    ].copy()

    if "categories" in classified.columns:
        for _, row in top20.iterrows():
            sub = row["submission_number"]
            dev_reports = classified[classified["matched_submission"] == sub]
            dev_classified = dev_reports[dev_reports["primary_category"].notna()]
            if len(dev_classified) == 0:
                continue
            cats = dev_classified["primary_category"].value_counts(normalize=True)
            if len(cats) > 0:
                top20.loc[top20["submission_number"] == sub, "primary_failure"] = cats.index[0]
                top20.loc[top20["submission_number"] == sub, "ai_specific_pct"] = round(
                    dev_classified["ai_specific"].mean() * 100, 1
                )

    top20.to_csv(TABLES_DIR / "top20_devices.csv", index=False)
    print("\n=== Top 20 Devices by Report Count ===")
    print(top20[["submission_number", "device_name", "panel", "report_count"]].to_string(index=False))
    return top20


def temporal_analysis(devices, linked, classified):
    """Reports per year, clearances per year, and failure mode trends."""
    reports_by_year = (
        linked[linked["report_year"].between(2020, 2026)]
        .groupby("report_year")
        .agg(n_reports=("report_number", "nunique"))
        .reset_index()
    )

    clearances_by_year = (
        devices[devices["clearance_year"].between(2015, 2026)]
        .groupby("clearance_year")
        .agg(n_clearances=("submission_number", "count"))
        .reset_index()
    )
    clearances_by_year["cumulative_devices"] = clearances_by_year["n_clearances"].cumsum()

    temporal = reports_by_year.merge(
        clearances_by_year,
        left_on="report_year",
        right_on="clearance_year",
        how="outer",
    )
    temporal["year"] = temporal["report_year"].fillna(temporal["clearance_year"]).astype(int)
    temporal = temporal.sort_values("year")

    if "categories" in classified.columns and "report_year" not in classified.columns:
        classified["date_received"] = pd.to_datetime(classified["date_received"], errors="coerce")
        classified["report_year"] = classified["date_received"].dt.year

    if "categories" in classified.columns:
        cat_by_year = []
        for year in range(2020, 2027):
            yr_data = classified[classified["report_year"] == year]
            if len(yr_data) == 0:
                continue
            row = {"year": year, "n": len(yr_data)}
            for cat in CATEGORIES:
                row[f"pct_{cat}"] = round(
                    yr_data["primary_category"].eq(cat).mean() * 100, 1
                )
            row["ai_specific_pct"] = round(yr_data["ai_specific"].mean() * 100, 1)
            cat_by_year.append(row)
        cat_temporal = pd.DataFrame(cat_by_year)
        temporal = temporal.merge(cat_temporal, on="year", how="left")

    temporal.to_csv(TABLES_DIR / "temporal_trends.csv", index=False)
    print("\n=== Temporal Trends ===")
    cols = ["year", "n_reports", "n_clearances", "cumulative_devices"]
    print(temporal[[c for c in cols if c in temporal.columns]].to_string(index=False))
    return temporal


def clearance_lag_analysis(device_counts, linked):
    """Time from clearance to first MAUDE report."""
    devices_with_reports = device_counts[device_counts["report_count"] > 0].copy()

    first_report = (
        linked.groupby("matched_submission")["date_received"]
        .min()
        .reset_index()
    )
    first_report.columns = ["submission_number", "first_report_date"]

    lag = devices_with_reports.merge(first_report, on="submission_number", how="left")
    lag["first_report_date"] = pd.to_datetime(lag["first_report_date"])
    lag["lag_days"] = (lag["first_report_date"] - lag["clearance_date"]).dt.days

    valid = lag[lag["lag_days"].notna() & (lag["lag_days"] >= 0)]
    print(f"\n=== Clearance-to-First-Report Lag ===")
    print(f"N devices with valid lag: {len(valid)}")
    print(f"Median lag: {valid['lag_days'].median():.0f} days ({valid['lag_days'].median()/365:.1f} years)")
    print(f"Mean lag: {valid['lag_days'].mean():.0f} days ({valid['lag_days'].mean()/365:.1f} years)")
    print(f"Min: {valid['lag_days'].min():.0f} days, Max: {valid['lag_days'].max():.0f} days")

    return lag


def manufacturer_analysis(device_counts):
    """Surveillance rates by manufacturer (consolidated)."""
    by_company = (
        device_counts.groupby("company_clean")
        .agg(
            n_devices=("submission_number", "count"),
            n_with_reports=("report_count", lambda x: (x > 0).sum()),
            total_reports=("report_count", "sum"),
        )
        .reset_index()
    )
    by_company["zero_report_pct"] = round(
        (1 - by_company["n_with_reports"] / by_company["n_devices"]) * 100, 1
    )
    by_company = by_company.sort_values("n_devices", ascending=False)

    print("\n=== Top 15 Manufacturers by Device Count ===")
    print(by_company.head(15).to_string(index=False))
    return by_company


def main():
    print("Loading data...")
    devices, linked, classified = load_data()

    device_counts = device_report_counts(devices, linked)
    specialty_analysis(device_counts, classified)
    top_20_devices(device_counts, classified)
    temporal_analysis(devices, linked, classified)
    clearance_lag_analysis(device_counts, linked)
    manufacturer_analysis(device_counts)

    print("\n=== Failure Mode Distribution (All Classified Reports) ===")
    if "categories" in classified.columns:
        progress = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")
        print(f"Total classified: {len(progress):,}")
        print(f"\nPrimary category:")
        print(progress["primary_category"].value_counts().to_string())
        all_cats = progress["categories"].str.split(",").explode()
        print(f"\nAll categories (multi-label):")
        print(all_cats.value_counts().to_string())
        print(f"\nAI-specific: {progress['ai_specific'].sum():,} / {len(progress):,} ({progress['ai_specific'].mean()*100:.1f}%)")

    print(f"\nAll tables saved to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
