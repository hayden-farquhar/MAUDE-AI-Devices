#!/usr/bin/env python3
"""Download and clean the FDA AI/ML-Enabled Medical Device List."""

from pathlib import Path
import io
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_URL = "https://www.fda.gov/media/178540/download?attachment"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def download_device_list() -> Path:
    """Download the Excel file and return the local path."""
    dest = RAW_DIR / "fda_aiml_devices.xlsx"
    print(f"Downloading FDA AI/ML device list from {EXCEL_URL}")
    r = requests.get(EXCEL_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  Saved {len(r.content):,} bytes → {dest.relative_to(PROJECT_ROOT)}")
    return dest


def parse_and_clean(xlsx_path: Path) -> pd.DataFrame:
    """Parse the Excel file into a clean DataFrame."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    col_map = {
        "Date of Final Decision": "clearance_date",
        "Submission Number": "submission_number",
        "Device": "device_name",
        "Company": "company",
        "Panel (Lead)": "panel",
        "Primary Product Code": "product_code",
    }
    keep = [c for c in col_map if c in df.columns]
    df = df[keep].rename(columns=col_map).copy()

    # Drop the hyperlink formula column if present
    if "submission" in [c.lower() for c in df.columns]:
        df = df.drop(columns=[c for c in df.columns if c.lower() == "submission"], errors="ignore")

    df.loc[:, "clearance_date"] = pd.to_datetime(df["clearance_date"], format="%m/%d/%Y", errors="coerce")
    df.loc[:, "submission_type"] = df["submission_number"].str.extract(r"^([A-Z]+)", expand=False)
    df.loc[:, "company_clean"] = (
        df["company"]
        .str.upper()
        .str.strip()
        .str.replace(r"[,.]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    print(f"  Parsed {len(df):,} devices, {df['product_code'].nunique()} unique product codes")
    print(f"  Panels: {df['panel'].value_counts().head(5).to_dict()}")
    print(f"  Submission types: {df['submission_type'].value_counts().to_dict()}")
    print(f"  Date range: {df['clearance_date'].min():%Y-%m-%d} to {df['clearance_date'].max():%Y-%m-%d}")
    return df


def main():
    xlsx_path = download_device_list()
    df = parse_and_clean(xlsx_path)

    out_path = PROCESSED_DIR / "fda_aiml_devices.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved cleaned CSV → {out_path.relative_to(PROJECT_ROOT)}")

    out_parquet = PROCESSED_DIR / "fda_aiml_devices.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"  Saved Parquet → {out_parquet.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
