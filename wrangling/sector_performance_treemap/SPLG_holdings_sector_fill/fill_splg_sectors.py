#!/usr/bin/env python3
"""
Fill missing 'sector' values for SPLG holdings using Yahoo Finance (yfinance).

Usage:
    python fill_splg_sectors.py --in holdings-daily-us-en-splg.xlsx --out holdings-with-sectors.xlsx
    # Or CSV -> CSV:
    python fill_splg_sectors.py --in holdings.csv --out holdings-with-sectors.csv
"""
import argparse
import os
import sys
import time
import json
from typing import Dict, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("This script requires 'yfinance'. Install with: pip install yfinance>=0.2.50", file=sys.stderr)
    sys.exit(1)

# --- Helpers ---
YF_TICKER_FIXES = {
    # Known Yahoo Finance dot-class tickers -> dash-class
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "HEI.A": "HEI-A",
    "NWS.A": "NWS-A",
    "NWS.B": "NWS-B",
    "FOXA": "FOXA",  # explicit in case class tickers are present
    "FOX": "FOX",
}

def normalize_ticker(t: str) -> str:
    t = (t or "").strip()
    return YF_TICKER_FIXES.get(t, t)

def load_cache(cache_path: str) -> Dict[str, str]:
    if os.path.exists(cache_path):
        try:
            return json.load(open(cache_path, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(cache: Dict[str, str], cache_path: str) -> None:
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, cache_path)

def fetch_sector_yf(ticker: str, max_retries: int = 3, pause: float = 0.7) -> Optional[str]:
    """
    Fetch sector from Yahoo Finance. Returns None if unavailable.
    Retries a few times to be resilient to transient errors / rate limits.
    """
    tfix = normalize_ticker(ticker)
    for attempt in range(1, max_retries + 1):
        try:
            tk = yf.Ticker(tfix)
            info = tk.get_info()
            sector = info.get("sector") or info.get("industry")  # sector preferred; industry as last resort
            if sector:
                return str(sector).strip()
        except Exception as e:
            # backoff a bit on error
            time.sleep(pause * attempt)
        time.sleep(pause)
    return None

def infer_sector_from_name(name: str) -> Optional[str]:
    """
    Very light heuristic fallbacks for a few mega-caps if YF fails.
    (You can extend this dictionary as needed.)
    """
    if not name:
        return None
    key = name.lower()
    rules = [
        ("financial", "Financials"),
        ("bank", "Financials"),
        ("pharma", "Health Care"),
        ("pharmaceutical", "Health Care"),
        ("health", "Health Care"),
        ("energy", "Energy"),
        ("oil", "Energy"),
        ("gas", "Energy"),
        ("utility", "Utilities"),
        ("software", "Information Technology"),
        ("semiconductor", "Information Technology"),
        ("chip", "Information Technology"),
        ("retail", "Consumer Discretionary"),
        ("consumer", "Consumer Staples"), # very rough; prefer YF
        ("industrial", "Industrials"),
        ("realty", "Real Estate"),
        ("reit", "Real Estate"),
        ("telecom", "Communication Services"),
        ("communications", "Communication Services"),
        ("media", "Communication Services"),
        ("materials", "Materials"),
    ]
    for kw, sector in rules:
        if kw in key:
            return sector
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV/XLSX file path")
    ap.add_argument("--out", dest="out_path", required=True, help="Output CSV/XLSX file path")
    ap.add_argument("--cache", dest="cache_path", default="sector_cache.json", help="Cache file to avoid refetching")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for testing (only process first N rows)")
    ap.add_argument("--pause", type=float, default=0.6, help="Sleep seconds between API calls to be gentle")
    args = ap.parse_args()

    in_ext = os.path.splitext(args.in_path)[1].lower()
    if in_ext in (".xlsx", ".xls"):
        df = pd.read_excel(args.in_path)
    elif in_ext in (".csv", ".txt"):
        df = pd.read_csv(args.in_path)
    else:
        print(f"Unsupported input extension: {in_ext}. Use .csv or .xlsx", file=sys.stderr)
        sys.exit(2)

    # Normalize required columns
    # We expect at least columns: 'ticker', 'sector' (with '-' to fill), optionally 'name'
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        print("Input must contain a 'ticker' column.", file=sys.stderr)
        sys.exit(3)
    if "sector" not in cols:
        # Create sector if missing
        df["sector"] = "-"
        cols = {c.lower(): c for c in df.columns}

    ticker_col = cols["ticker"]
    sector_col = cols["sector"]
    name_col = cols.get("name", None)

    cache = load_cache(args.cache_path)

    # Work set
    work_df = df if args.limit is None else df.head(args.limit)

    n_total = len(work_df)
    n_done = 0
    n_filled = 0

    for idx, row in work_df.iterrows():
        current_sector = str(row.get(sector_col, "")).strip()
        if current_sector and current_sector != "-":
            n_done += 1
            continue

        ticker = str(row[ticker_col]).strip()
        # Check cache
        if ticker in cache:
            sec = cache[ticker]
        else:
            sec = fetch_sector_yf(ticker)
            # Fallback: try by name if no sector
            if not sec and name_col is not None:
                sec = infer_sector_from_name(str(row[name_col]))
            # Final fallback: leave '-' if still unknown
            cache[ticker] = sec if sec else "-"
            save_cache(cache, args.cache_path)
            time.sleep(args.pause)

        if sec and sec != "-":
            df.at[idx, sector_col] = sec
            n_filled += 1

        n_done += 1
        if n_done % 25 == 0 or n_done == n_total:
            print(f"[{n_done}/{n_total}] filled so far: {n_filled}")

    # Save output
    out_ext = os.path.splitext(args.out_path)[1].lower()
    if out_ext in (".xlsx", ".xls"):
        df.to_excel(args.out_path, index=False)
    elif out_ext in (".csv", ".txt"):
        df.to_csv(args.out_path, index=False)
    else:
        print(f"Unsupported output extension: {out_ext}. Use .csv or .xlsx", file=sys.stderr)
        sys.exit(4)

    print(f"Done. Filled {n_filled} sector values. Wrote: {args.out_path}")
    print(f"Cache at: {args.cache_path}")

if __name__ == "__main__":
    main()
