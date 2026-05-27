#!/usr/bin/env python3
"""Query openFDA MAUDE /device/event for all AI device product codes.

Handles openFDA's skip+limit <= 26,000 constraint by date-partitioning
high-volume product codes into yearly/quarterly windows.

Outputs one JSONL file per product code under data/raw/maude_by_code/.
Saves a summary CSV of per-code report counts.

Resumable: skips product codes whose JSONL already exists on disk.
"""

from pathlib import Path
import json
import time
import requests
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MAUDE_DIR = PROJECT_ROOT / "data" / "raw" / "maude_by_code"
MAUDE_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.fda.gov/device/event.json"
PAGE_SIZE = 500  # 1000 requires API key; 500 works without one
MAX_SKIP = 25500  # openFDA cap: skip + limit <= 26000
DATE_START = "20200101"
DATE_END = "20260601"
REQ_DELAY = 0.35  # ~2.8 req/s, conservative to avoid rate limits

YEARLY_WINDOWS = [
    ("20200101", "20201231"),
    ("20210101", "20211231"),
    ("20220101", "20221231"),
    ("20230101", "20231231"),
    ("20240101", "20241231"),
    ("20250101", "20251231"),
    ("20260101", "20260601"),
]

QUARTERLY_WINDOWS = []
for year in range(2020, 2027):
    for q_start, q_end in [("0101", "0331"), ("0401", "0630"), ("0701", "0930"), ("1001", "1231")]:
        start = f"{year}{q_start}"
        end = f"{year}{q_end}"
        if int(start) > int(DATE_END):
            break
        end = min(end, DATE_END)
        QUARTERLY_WINDOWS.append((start, end))


MAX_RETRIES = 4
RETRY_BACKOFF = [2, 5, 15, 30]


def _request_with_retry(params: dict) -> requests.Response:
    """Make an openFDA request with exponential backoff on 403/429."""
    for attempt in range(MAX_RETRIES + 1):
        r = requests.get(API_BASE, params=params, timeout=30)
        if r.status_code in (403, 429) and attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF[attempt]
            time.sleep(wait)
            continue
        return r
    return r


def get_count(search_query: str) -> int:
    """Return total result count for a search query."""
    r = _request_with_retry({"search": search_query, "limit": 0})
    if r.status_code == 404:
        return 0
    r.raise_for_status()
    return r.json().get("meta", {}).get("results", {}).get("total", 0)


def paginate(search_query: str, max_results: int | None = None) -> list[dict]:
    """Paginate through results for a query, respecting the 26K skip cap."""
    results = []
    skip = 0
    while True:
        if skip > MAX_SKIP:
            break
        params = {"search": search_query, "limit": PAGE_SIZE, "skip": skip}
        r = _request_with_retry(params)
        time.sleep(REQ_DELAY)
        if r.status_code == 404:
            break
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        results.extend(batch)
        if max_results and len(results) >= max_results:
            break
        if len(batch) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
    return results


def fetch_code(product_code: str) -> list[dict]:
    """Fetch all MAUDE reports for a product code, date-partitioning if needed."""
    base_query = (
        f'device.device_report_product_code:"{product_code}" '
        f"AND date_received:[{DATE_START} TO {DATE_END}]"
    )
    total = get_count(base_query)
    time.sleep(REQ_DELAY)

    if total == 0:
        return []

    # Fits within single pagination window
    if total <= MAX_SKIP + PAGE_SIZE:
        return paginate(base_query)

    # Partition by year
    all_results = []
    for y_start, y_end in YEARLY_WINDOWS:
        year_query = (
            f'device.device_report_product_code:"{product_code}" '
            f"AND date_received:[{y_start} TO {y_end}]"
        )
        year_count = get_count(year_query)
        time.sleep(REQ_DELAY)

        if year_count == 0:
            continue

        if year_count <= MAX_SKIP + PAGE_SIZE:
            all_results.extend(paginate(year_query))
        else:
            # Further partition by quarter
            for q_start, q_end in QUARTERLY_WINDOWS:
                if not (q_start >= y_start and q_end <= y_end):
                    continue
                q_query = (
                    f'device.device_report_product_code:"{product_code}" '
                    f"AND date_received:[{q_start} TO {q_end}]"
                )
                q_results = paginate(q_query)
                all_results.extend(q_results)

    # Deduplicate by report_number
    seen = set()
    deduped = []
    for rec in all_results:
        rn = rec.get("report_number", "")
        if rn and rn in seen:
            continue
        seen.add(rn)
        deduped.append(rec)
    return deduped


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def main():
    device_path = PROCESSED_DIR / "fda_aiml_devices.parquet"
    if not device_path.exists():
        device_path = PROCESSED_DIR / "fda_aiml_devices.csv"
    if not device_path.exists():
        raise FileNotFoundError("Run 01_download_device_list.py first")

    df = pd.read_parquet(device_path) if device_path.suffix == ".parquet" else pd.read_csv(device_path)
    codes = sorted(df["product_code"].dropna().unique())
    print(f"Querying openFDA MAUDE for {len(codes)} product codes (2020–2026)")

    summary = []
    for code in tqdm(codes, desc="Product codes"):
        out_path = MAUDE_DIR / f"{code}.jsonl"
        if out_path.exists():
            n = sum(1 for _ in open(out_path))
            summary.append({"product_code": code, "maude_reports": n, "status": "cached"})
            continue

        try:
            records = fetch_code(code)
            save_jsonl(records, out_path)
            summary.append({"product_code": code, "maude_reports": len(records), "status": "ok"})
        except Exception as e:
            summary.append({"product_code": code, "maude_reports": -1, "status": f"error: {e}"})
            tqdm.write(f"  ERROR {code}: {e}")

    summary_df = pd.DataFrame(summary)
    summary_path = PROCESSED_DIR / "maude_query_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    ok = summary_df[summary_df["maude_reports"] >= 0]
    with_reports = ok[ok["maude_reports"] > 0]
    total_reports = ok["maude_reports"].sum()
    print(f"\nDone. {len(with_reports)}/{len(ok)} codes have MAUDE reports.")
    print(f"Total reports downloaded: {total_reports:,.0f}")
    print(f"Summary → {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
