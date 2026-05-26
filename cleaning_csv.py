# -*- coding: utf-8 -*-
"""
clean_csv.py
Cleans the Memoni video links CSV:
  - Fetches from GitHub (or reads a local file)
  - Drops rows with duplicate or empty videolinks
  - Fills empty platform values from the URL domain
  - Filters rows: keeps only those where post_text is empty/NaN
    OR contains 'memon'/'memoni'
  - Drops the post_text column from the output
  - Saves the result as cleaned_videos.csv

Usage:
  # Fetch from GitHub:
  python clean_csv.py

  # Use a local CSV:
  python clean_csv.py --csv-path /path/to/videos_links.csv

  # Custom output path:
  python clean_csv.py --output /path/to/cleaned_videos.csv
"""

import re
import argparse
from io import StringIO
from urllib.parse import urlparse

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/hamzahanif2210/memoni_audio"
    "/refs/heads/main/videos_links.csv"
)
MEMON_PATTERN = re.compile(r'#?memoni?\b', re.IGNORECASE)

_PLATFORM_MAP = {
    "youtube.com":   "youtube",
    "youtu.be":      "youtube",
    "instagram.com": "instagram",
    "tiktok.com":    "tiktok",
}


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csv_from_github(url: str) -> pd.DataFrame:
    print(f"Fetching CSV from: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df.columns = df.columns.str.strip()
    required_cols = {"videolink", "category", "channel_name", "post_text", "platform"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    print(f"Loaded {len(df)} rows.  Columns: {df.columns.tolist()}")
    return df


def drop_duplicate_links(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["videolink"]).reset_index(drop=True)
    dropped_nan = before - len(df)
    if dropped_nan:
        print(f"Dropped {dropped_nan} row(s) with empty videolink.")

    dupes = df[df.duplicated(subset="videolink", keep=False)]
    if dupes.empty:
        print("No duplicate links found.")
        return df

    dup_links = df[df.duplicated(subset="videolink", keep="first")]["videolink"]
    print(f"Found {len(dup_links)} duplicate row(s) — removing:")
    for link in dup_links.values:
        print(f"  DUPLICATE (removed): {str(link)[:80]}")

    return df.drop_duplicates(subset="videolink", keep="first").reset_index(drop=True)


def infer_platform(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        for domain, name in _PLATFORM_MAP.items():
            if host == domain or host.endswith("." + domain):
                return name
    except Exception:
        pass
    return ""


def fill_platform(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["platform"].isna() | (df["platform"].str.strip() == "")
    if mask.any():
        df = df.copy()
        df.loc[mask, "platform"] = df.loc[mask, "videolink"].apply(infer_platform)
        print(f"Platform column: filled {mask.sum()} empty value(s) from URL.")
    else:
        print("Platform column: no empty values.")
    return df


def should_process_video(row: pd.Series) -> bool:
    text = row.get("post_text", "")
    if pd.isna(text) or str(text).strip() == "":
        return True
    return bool(MEMON_PATTERN.search(str(text)))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clean Memoni video links CSV")
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help="Path to a local CSV file (skips GitHub fetch)"
    )
    parser.add_argument(
        "--output", type=str, default="cleaned_videos.csv",
        help="Output path for the cleaned CSV (default: cleaned_videos.csv)"
    )
    args = parser.parse_args()

    # 1. Load
    if args.csv_path:
        print(f"Reading local CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        df.columns = df.columns.str.strip()
    else:
        df = load_csv_from_github(CSV_URL)

    total_rows = len(df)

    # 2. Deduplicate
    df = drop_duplicate_links(df)

    # 3. Fill platform
    df = fill_platform(df)

    # 4. Filter by memon/memoni keyword
    df_filtered = df[df.apply(should_process_video, axis=1)].reset_index(drop=True)
    skipped = total_rows - len(df_filtered)
    print(
        f"\nKeyword filter: {len(df_filtered)}/{total_rows} rows pass "
        f"(skipped {skipped} without memon/memoni in post_text)"
    )

    # 5. Drop post_text column
    df_filtered = df_filtered.drop(columns=["post_text"], errors="ignore")

    # 6. Save
    df_filtered.to_csv(args.output, index=False)
    print(f"\nCleaned CSV saved: {args.output}  ({len(df_filtered)} rows)")
    print(f"Columns: {df_filtered.columns.tolist()}")


if __name__ == "__main__":
    main()