#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FUREcast Level-4 Drill-Down Dataset Builder
- Loads SPLG holdings CSV (from SSGA, data fresh as of 10-Oct-2025) with Ticker/Company/Sector/Weight
- Fetches per-stock daily % change (last close vs prior close) for ALL tickers
- Optionally enriches with KPIs (PE, DividendYield, Beta) using yfinance
- Outputs a single CSV for Plotly treemap drill-down:
    data_out/level4_treemap_nodes.csv

Treemap usage:
  path   = ["Sector", "Company"]
  values = "Weight (%)"
  color  = "DailyChangePct"
  hover  = ["Weight (%)", "PE", "Beta", "DividendYield"]
"""

import os
import io
import time
import math
import traceback
from typing import Dict, List

import pandas as pd

# ----------------- Config -----------------
DATA_IN = "../../data/holdings-with-sectors.csv"
# DATA_IN = os.path.join("../../../data", "/holdings-with-sectors.xlsx")
# DATA_IN = os.path.join("data_in", "splg_holdings.csv")
DATA_OUT_DIR = "data_out"
os.makedirs(DATA_OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(DATA_OUT_DIR, "level4_treemap_nodes.csv")

ENABLE_KPI_ENRICHMENT = True        # set False if you want a faster run
KPI_SLEEP_SECS = 0.15               # be gentle to Yahoo endpoints

# If you want to limit tickers for testing (e.g., first 50)
LIMIT_TICKERS = None                # e.g., 50

# -----------------------------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    return d

def load_holdings(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Holdings CSV not found at {csv_path}.\n"
            "Download from SSGA SPLG fund page → Holdings → CSV and save as data_in/splg_holdings.csv"
        )
    df = pd.read_csv(csv_path)
    df = normalize_cols(df)

    # Map columns commonly found in SSGA CSVs
    colmap = {
        "ticker": None,
        "company": None,
        "sector": None,
        "weight_pct": None
    }
    # Try to detect relevant columns
    candidates = {
        "ticker": ["ticker", "symbol"],
        "company": ["name", "company", "security_name", "holding"],
        "sector": ["sector", "gics_sector"],
        "weight_pct": ["weight", "weight_%", "weighting", "weight_%_of_fund"]
    }
    for k, choices in candidates.items():
        for c in choices:
            if c in df.columns:
                colmap[k] = c
                break

    missing = [k for k, v in colmap.items() if v is None and k in ("ticker", "company", "sector")]
    if missing:
        raise ValueError(f"Missing required columns in holdings CSV: {missing}\nGot columns: {list(df.columns)}")

    out = pd.DataFrame({
        "Ticker": df[colmap["ticker"]].astype(str).str.upper(),
        "Company": df[colmap["company"]],
        "Sector": df[colmap["sector"]].astype(str).str.strip()
    })
    # weight is optional but strongly recommended
    if colmap["weight_pct"] is not None:
        out["Weight (%)"] = pd.to_numeric(df[colmap["weight_pct"]], errors="coerce")
    else:
        out["Weight (%)"] = pd.NA

    out = out.dropna(subset=["Ticker", "Sector"]).reset_index(drop=True)
    return out

def fetch_daily_pct_change_for_all(tickers: List[str]) -> pd.DataFrame:
    """
    Uses yfinance to download last 2 daily closes for ALL tickers at once,
    then computes % change. Robust to some tickers missing data.
    """
    import yfinance as yf

    # yfinance can handle long lists via download; we’ll batch to be safe
    batch_size = 150  # conservative; adjust upward if you want
    frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        # `download` returns a panel-like structure. With `group_by='ticker'`, columns are per ticker.
        data = yf.download(
            tickers=batch,
            period="5d",           # a few days to survive holidays
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True
        )

        # Two structures possible: multi-index columns when multiple tickers,
        # or a simple frame if batch has one ticker.
        # We'll normalize into long form per ticker with last two closes.
        if isinstance(data.columns, pd.MultiIndex):
            tickers_in_batch = sorted({t for (t, col) in data.columns if col == "Close"})
            for tk in tickers_in_batch:
                closes = data[tk]["Close"].dropna()
                if len(closes) >= 2:
                    last = closes.iloc[-1]
                    prev = closes.iloc[-2]
                    pct = ((last - prev) / prev) * 100.0 if prev else None
                elif len(closes) == 1:
                    last = closes.iloc[-1]
                    prev = None
                    pct = None
                else:
                    last = prev = pct = None
                frames.append(pd.DataFrame({"Ticker":[tk], "DailyChangePct":[pct]}))
        else:
            # Single ticker form
            closes = data["Close"].dropna() if "Close" in data.columns else pd.Series(dtype=float)
            tk = batch[0]
            if len(closes) >= 2:
                last = closes.iloc[-1]
                prev = closes.iloc[-2]
                pct = ((last - prev) / prev) * 100.0 if prev else None
            elif len(closes) == 1:
                pct = None
            else:
                pct = None
            frames.append(pd.DataFrame({"Ticker":[tk], "DailyChangePct":[pct]}))

        time.sleep(0.1)  # be nice

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["Ticker","DailyChangePct"])
    # In case duplicates appear (rare), keep last
    out = out.drop_duplicates(subset=["Ticker"], keep="last").reset_index(drop=True)
    return out

def enrich_kpis_yfinance(df_ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Adds PE, DividendYield, Beta using yfinance .info (best-effort).
    """
    if not ENABLE_KPI_ENRICHMENT:
        return df_ticks.assign(PE=pd.NA, DividendYield=pd.NA, Beta=pd.NA)

    import yfinance as yf
    df = df_ticks.copy()
    df["PE"] = pd.NA; df["DividendYield"] = pd.NA; df["Beta"] = pd.NA

    for i, tk in enumerate(df["Ticker"].unique()):
        try:
            info = yf.Ticker(tk).info
            pe   = info.get("trailingPE")
            dy   = info.get("dividendYield")
            beta = info.get("beta")
            df.loc[df["Ticker"] == tk, ["PE","DividendYield","Beta"]] = [pe, dy, beta]
        except Exception:
            # leave NaNs
            pass
        time.sleep(KPI_SLEEP_SECS)

    return df

def main():
    print("=== FUREcast Level-4 dataset builder ===")
    # 1) Load holdings
    holdings = load_holdings(DATA_IN)
    if LIMIT_TICKERS:
        holdings = holdings.iloc[:LIMIT_TICKERS].copy()
    print(f"Holdings rows: {len(holdings)}")

    # 2) Normalize weights (optional but nice)
    if "Weight (%)" in holdings.columns:
        total = pd.to_numeric(holdings["Weight (%)"], errors="coerce").sum()
        if pd.notna(total) and total > 0:
            holdings["Weight (%)"] = pd.to_numeric(holdings["Weight (%)"], errors="coerce") * (100.0 / total)

    # 3) Fetch per-stock daily % change (vectorized batches)
    tickers = holdings["Ticker"].dropna().unique().tolist()
    print(f"Fetching per-stock daily % change for {len(tickers)} tickers...")
    df_chg = fetch_daily_pct_change_for_all(tickers)

    # 4) Optional KPI enrichment
    print("Enriching with KPIs (PE, DividendYield, Beta)...")
    df_kpi = enrich_kpis_yfinance(pd.DataFrame({"Ticker": tickers}))

    # 5) Merge: holdings + change + KPI
    df = holdings.merge(df_chg, on="Ticker", how="left").merge(df_kpi, on="Ticker", how="left")

    # 6) Clean/guard
    df["DailyChangePct"] = pd.to_numeric(df["DailyChangePct"], errors="coerce")
    df["Weight (%)"] = pd.to_numeric(df["Weight (%)"], errors="coerce")
    # If some tiles have zero/NaN weight, give tiny epsilon so Plotly can render
    df["Weight (%)"] = df["Weight (%)"].fillna(0.0001).clip(lower=0.0001)

    # 7) Save
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved treemap nodes to: {OUT_CSV}")

    # 8) Quick summary
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

"""
Terminal output when this script was run to create the treemap dataset:

12:58:01|robert@SLS-2:~/FEURCast/wrangling/sector_performace_treemap$ python3 build_treemap_dataset.py 
=== FUREcast Level-4 dataset builder ===
ERROR: 'utf-8' codec can't decode bytes in position 15-16: invalid continuation byte
Traceback (most recent call last):
  File "/home/robert/FEURCast/wrangling/sector_performace_treemap/build_treemap_dataset.py", line 224, in <module>
    main()
  File "/home/robert/FEURCast/wrangling/sector_performace_treemap/build_treemap_dataset.py", line 186, in main
    holdings = load_holdings(DATA_IN)
               ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/robert/FEURCast/wrangling/sector_performace_treemap/build_treemap_dataset.py", line 55, in load_holdings
    df = pd.read_csv(csv_path)
         ^^^^^^^^^^^^^^^^^^^^^
  File "/home/robert/anaconda3/envs/feurcast-env/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/robert/anaconda3/envs/feurcast-env/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/robert/anaconda3/envs/feurcast-env/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/robert/anaconda3/envs/feurcast-env/lib/python3.11/site-packages/pandas/io/parsers/readers.py", line 1898, in _make_engine
    return mapping[engine](f, **self.options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/robert/anaconda3/envs/feurcast-env/lib/python3.11/site-packages/pandas/io/parsers/c_parser_wrapper.py", line 93, in __init__
    self._reader = parsers.TextReader(src, **kwds)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/parsers.pyx", line 574, in pandas._libs.parsers.TextReader.__cinit__
  File "pandas/_libs/parsers.pyx", line 663, in pandas._libs.parsers.TextReader._get_header
  File "pandas/_libs/parsers.pyx", line 874, in pandas._libs.parsers.TextReader._tokenize_rows
  File "pandas/_libs/parsers.pyx", line 891, in pandas._libs.parsers.TextReader._check_tokenize_status
  File "pandas/_libs/parsers.pyx", line 2053, in pandas._libs.parsers.raise_parser_error
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 15-16: invalid continuation byte

"""