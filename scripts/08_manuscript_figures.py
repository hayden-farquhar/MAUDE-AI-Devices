#!/usr/bin/env python3
"""Generate publication-quality figures and tables for manuscript.

Figures:
  1. Report-per-device distribution (histogram + zero-report callout)
  2. Failure mode distribution (horizontal bar chart, AI-specific vs generic)
  3. Temporal trends (dual-axis: clearances + reports per year)
  4. Specialty-stratified failure mode heatmap
  5. Sankey: device specialty → failure category → patient harm

Tables:
  1. Study cohort descriptive statistics
  2. Top 20 devices by report count
  3. Per-specialty surveillance rates
  4. Classifier validation metrics

Outputs saved to outputs/figures/ and outputs/tables/
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_LABELS = {
    "A": "Data input",
    "B": "Algorithmic/\nprocessing",
    "C": "Output/\ninterpretation",
    "D": "User\ninteraction",
    "E": "Infrastructure/\nintegration",
    "F": "Hardware",
    "G": "Patient harm",
    "U": "Uninformative",
}

CATEGORY_COLORS = {
    "A": "#4e79a7",
    "B": "#f28e2b",
    "C": "#e15759",
    "D": "#76b7b2",
    "E": "#59a14f",
    "F": "#af7aa1",
    "G": "#ff9da7",
    "U": "#bab0ac",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_data():
    devices = pd.read_csv(PROCESSED_DIR / "fda_aiml_devices.csv")
    devices["clearance_date"] = pd.to_datetime(devices["clearance_date"])
    devices["clearance_year"] = devices["clearance_date"].dt.year

    linked = pd.read_parquet(PROCESSED_DIR / "linked_reports.parquet")
    linked["date_received"] = pd.to_datetime(linked["date_received"], errors="coerce")
    linked["report_year"] = linked["date_received"].dt.year

    classified_path = PROCESSED_DIR / "classified_reports.parquet"
    if classified_path.exists():
        classified = pd.read_parquet(classified_path)
    else:
        progress = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")
        classified = linked.merge(progress, on="report_number", how="inner")

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

    return devices, linked, classified, device_counts


def figure1_report_distribution(device_counts):
    """Histogram of reports per device with zero-report callout."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

    n_zero = (device_counts["report_count"] == 0).sum()
    n_nonzero = (device_counts["report_count"] > 0).sum()
    n_total = len(device_counts)
    pct_zero = n_zero / n_total * 100

    ax1.bar(["Zero reports", "≥1 report"], [n_zero, n_nonzero],
            color=["#e15759", "#4e79a7"], edgecolor="white", width=0.6)
    ax1.set_ylabel("Number of AI/ML devices")
    ax1.set_title("A. MAUDE reporting coverage")
    ax1.text(0, n_zero + 20, f"{n_zero:,}\n({pct_zero:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax1.text(1, n_nonzero + 20, f"{n_nonzero:,}\n({100-pct_zero:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax1.set_ylim(0, n_zero * 1.15)
    ax1.spines[["top", "right"]].set_visible(False)

    nonzero = device_counts[device_counts["report_count"] > 0]["report_count"]
    bins = [1, 2, 5, 10, 20, 50, 100, 200, 500, nonzero.max() + 1]
    ax2.hist(nonzero, bins=bins, color="#4e79a7", edgecolor="white", alpha=0.9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of MAUDE reports")
    ax2.set_ylabel("Number of devices")
    ax2.set_title("B. Distribution among reported devices")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(w_pad=3)
    fig.savefig(FIGURES_DIR / "fig1_report_distribution.png")
    fig.savefig(FIGURES_DIR / "fig1_report_distribution.pdf")
    plt.close(fig)
    print("Figure 1 saved: report distribution")


def figure2_failure_modes(classified):
    """Horizontal bar chart: failure mode distribution, AI-specific overlay."""
    progress = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")

    cats = progress["categories"].str.split(",").explode()
    cat_counts = cats.value_counts()

    ai_only = progress[progress["ai_specific"] == True]
    ai_cats = ai_only["categories"].str.split(",").explode()
    ai_cat_counts = ai_cats.value_counts()

    order = ["B", "C", "A", "D", "F", "E", "G", "U"]
    order = [c for c in order if c in cat_counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))

    y_pos = np.arange(len(order))
    total_vals = [cat_counts.get(c, 0) for c in order]
    ai_vals = [ai_cat_counts.get(c, 0) for c in order]

    bars_total = ax.barh(y_pos, total_vals, height=0.6, color="#bab0ac", label="All reports", edgecolor="white")
    bars_ai = ax.barh(y_pos, ai_vals, height=0.6, color="#f28e2b", label="AI-specific", edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([CATEGORY_LABELS.get(c, c) for c in order])
    ax.set_xlabel("Number of reports (multi-label)")
    ax.set_title("Failure mode distribution across MAUDE reports for AI/ML devices")
    ax.legend(loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    for i, (total, ai) in enumerate(zip(total_vals, ai_vals)):
        ax.text(total + max(total_vals) * 0.01, i, f"{total:,}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_failure_modes.png")
    fig.savefig(FIGURES_DIR / "fig2_failure_modes.pdf")
    plt.close(fig)
    print("Figure 2 saved: failure mode distribution")


def figure3_temporal_trends(devices, linked):
    """Dual-axis: device clearances and MAUDE reports over time."""
    clearances = (
        devices[devices["clearance_year"].between(2015, 2025)]
        .groupby("clearance_year")
        .size()
        .reset_index(name="n_clearances")
    )

    reports = (
        linked[linked["report_year"].between(2020, 2025)]
        .groupby("report_year")
        .agg(n_reports=("report_number", "nunique"))
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color1 = "#4e79a7"
    ax1.bar(clearances["clearance_year"], clearances["n_clearances"],
            color=color1, alpha=0.7, width=0.7, label="Device clearances")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of AI/ML device clearances", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#e15759"
    ax2.plot(reports["report_year"], reports["n_reports"], "o-",
             color=color2, linewidth=2, markersize=6, label="MAUDE reports")
    ax2.set_ylabel("Number of linked MAUDE reports", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax1.set_title("AI/ML device clearances and MAUDE adverse event reports")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_temporal_trends.png")
    fig.savefig(FIGURES_DIR / "fig3_temporal_trends.pdf")
    plt.close(fig)
    print("Figure 3 saved: temporal trends")


def figure4_specialty_heatmap(classified):
    """Heatmap of failure modes by medical specialty."""
    top_panels = classified["matched_panel"].value_counts().head(8).index.tolist()
    subset = classified[classified["matched_panel"].isin(top_panels)]

    cats = ["B", "C", "A", "D", "F", "E", "G", "U"]
    matrix = pd.DataFrame(index=top_panels, columns=cats, dtype=float)

    for panel in top_panels:
        panel_data = subset[subset["matched_panel"] == panel]
        total = len(panel_data)
        if total == 0:
            continue
        for cat in cats:
            count = panel_data["categories"].str.contains(cat, na=False).sum()
            matrix.loc[panel, cat] = count / total * 100

    matrix = matrix.astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([CATEGORY_LABELS.get(c, c) for c in cats], rotation=45, ha="right")
    ax.set_yticks(range(len(top_panels)))
    ax.set_yticklabels(top_panels)

    for i in range(len(top_panels)):
        for j in range(len(cats)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                color = "white" if val > 40 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of reports (multi-label)")
    ax.set_title("Failure mode prevalence by medical specialty")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_specialty_heatmap.png")
    fig.savefig(FIGURES_DIR / "fig4_specialty_heatmap.pdf")
    plt.close(fig)
    print("Figure 4 saved: specialty heatmap")


def figure5_surveillance_by_clearance_year(device_counts):
    """Surveillance rate (% with ≥1 report) by clearance year cohort."""
    cohort = (
        device_counts[device_counts["clearance_year"].between(2016, 2025)]
        .groupby("clearance_year")
        .agg(
            n_devices=("submission_number", "count"),
            n_reported=("report_count", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    cohort["surveillance_pct"] = cohort["n_reported"] / cohort["n_devices"] * 100

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color1 = "#bab0ac"
    ax1.bar(cohort["clearance_year"], cohort["n_devices"],
            color=color1, alpha=0.7, width=0.7, label="Devices cleared")
    ax1.set_xlabel("Year of FDA clearance")
    ax1.set_ylabel("Number of devices cleared", color="black")

    ax2 = ax1.twinx()
    color2 = "#e15759"
    ax2.plot(cohort["clearance_year"], cohort["surveillance_pct"], "o-",
             color=color2, linewidth=2, markersize=6, label="% with ≥1 MAUDE report")
    ax2.set_ylabel("Surveillance rate (%)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 100)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax1.set_title("MAUDE surveillance rate by clearance year cohort")
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_surveillance_by_year.png")
    fig.savefig(FIGURES_DIR / "fig5_surveillance_by_year.pdf")
    plt.close(fig)
    print("Figure 5 saved: surveillance rate by clearance year")


def table1_cohort_descriptives(devices, linked, classified, device_counts):
    """Table 1: Study cohort descriptive statistics."""
    progress = pd.read_csv(PROCESSED_DIR / "classification_progress.csv")

    n_devices = len(devices)
    n_with_reports = (device_counts["report_count"] > 0).sum()
    n_reports = linked["report_number"].nunique()

    rows = [
        ("FDA AI/ML devices (total)", f"{n_devices:,}"),
        ("Submission type — 510(k)", f"{(devices['submission_type'] == 'K').sum():,} ({(devices['submission_type'] == 'K').mean()*100:.1f}%)"),
        ("Submission type — De Novo", f"{(devices['submission_type'] == 'DEN').sum():,} ({(devices['submission_type'] == 'DEN').mean()*100:.1f}%)"),
        ("Submission type — PMA", f"{(devices['submission_type'] == 'P').sum():,} ({(devices['submission_type'] == 'P').mean()*100:.1f}%)"),
        ("Unique product codes", f"{devices['product_code'].nunique()}"),
        ("Medical specialties", f"{devices['panel'].nunique()}"),
        ("Clearance date range", f"{devices['clearance_date'].min().strftime('%Y-%m-%d')} to {devices['clearance_date'].max().strftime('%Y-%m-%d')}"),
        ("", ""),
        ("Linked MAUDE reports", f"{n_reports:,}"),
        ("Devices with ≥1 MAUDE report", f"{n_with_reports:,} ({n_with_reports/n_devices*100:.1f}%)"),
        ("Devices with zero reports", f"{n_devices - n_with_reports:,} ({(n_devices - n_with_reports)/n_devices*100:.1f}%)"),
        ("Reports per device — median (IQR)", f"{device_counts['report_count'].median():.0f} ({device_counts['report_count'].quantile(0.25):.0f}–{device_counts['report_count'].quantile(0.75):.0f})"),
        ("Reports per device — mean (SD)", f"{device_counts['report_count'].mean():.1f} ({device_counts['report_count'].std():.1f})"),
        ("Narrative available", f"{(linked['narrative'].notna() & (linked['narrative'].str.len() > 50)).sum():,} ({(linked['narrative'].notna() & (linked['narrative'].str.len() > 50)).mean()*100:.1f}%)"),
        ("", ""),
        ("Classified reports", f"{len(progress):,}"),
        ("AI-specific failures", f"{progress['ai_specific'].sum():,} ({progress['ai_specific'].mean()*100:.1f}%)"),
        ("High confidence classifications", f"{(progress['confidence'] == 'high').sum():,} ({(progress['confidence'] == 'high').mean()*100:.1f}%)"),
    ]

    table = pd.DataFrame(rows, columns=["Characteristic", "Value"])
    table.to_csv(TABLES_DIR / "table1_cohort.csv", index=False)
    print("\nTable 1 saved: cohort descriptives")
    return table


def main():
    print("Loading data...")
    devices, linked, classified, device_counts = load_data()

    figure1_report_distribution(device_counts)
    figure2_failure_modes(classified)
    figure3_temporal_trends(devices, linked)
    figure4_specialty_heatmap(classified)
    figure5_surveillance_by_clearance_year(device_counts)
    table1_cohort_descriptives(devices, linked, classified, device_counts)

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    print(f"All tables saved to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
