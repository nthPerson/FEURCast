#!/usr/bin/env python3
"""
Quick yfinance diagnostics for SPLG → SPYM ticker change.

What it does:
- Fetches recent daily price history for both `SPLG` and `SPYM`.
- Prints row counts, date ranges, last available date, last close/volume.
- Highlights if data is empty or appears stale.

Run:
    python wrangling/check_yfinance_splg_spym.py

Requires:
    yfinance, pandas
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
import yfinance as yf


def fetch_summary(ticker: str, period: str = "6mo", interval: str = "1d") -> Dict[str, Any]:
    out: Dict[str, Any] = {"ticker": ticker, "ok": False, "error": None}
    try:
        tkr = yf.Ticker(ticker)
        df = tkr.history(period=period, interval=interval, auto_adjust=False)
        out["empty"] = df.empty
        out["rows"] = 0 if df.empty else int(df.shape[0])
        if not df.empty:
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            start = df.index.min()
            end = df.index.max()
            last_row = df.iloc[-1]
            out.update({
                "start": start.isoformat(),
                "end": end.isoformat(),
                "last_close": float(last_row.get("Close", float("nan"))),
                "last_volume": int(last_row.get("Volume", 0)) if pd.notna(last_row.get("Volume", pd.NA)) else None,
            })
            # Basic staleness heuristic: last bar older than 5 calendar days
            staleness_days = (datetime.utcnow() - end.to_pydatetime().replace(tzinfo=None)).days
            out["stale_days"] = staleness_days
            out["stale"] = staleness_days > 5
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def format_summary(s: Dict[str, Any]) -> str:
    if not s.get("ok"):
        return f"{s['ticker']}: ERROR -> {s.get('error')}"
    if s.get("empty"):
        return f"{s['ticker']}: EMPTY (no rows)"
    stale_flag = " (STALE)" if s.get("stale") else ""
    parts = [
        f"{s['ticker']}: {s['rows']} rows {stale_flag}",
        f"  Range: {s['start']} → {s['end']}",
        f"  Last close: {s['last_close']:.4f}  Volume: {s['last_volume']}",
    ]
    if s.get("stale"):
        parts.append(f"  Note: last bar is {s['stale_days']} days old")
    return "\n".join(parts)


def main() -> int:
    print("\n" + "=" * 80)
    print("  yfinance SPLG vs SPYM Diagnostic")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)

    tickers = ["SPLG", "SPYM"]
    results = {t: fetch_summary(t) for t in tickers}

    print()
    for t in tickers:
        print(format_summary(results[t]))
        print()

    # Guidance summary
    splg = results.get("SPLG", {})
    spym = results.get("SPYM", {})

    print("-" * 80)
    if splg.get("ok") and not splg.get("empty"):
        print("SPLG returned data.")
        if splg.get("stale"):
            print("However, SPLG data looks stale compared to current date.")
    else:
        print("SPLG is not returning usable data (empty or error).")

    if spym.get("ok") and not spym.get("empty"):
        print("SPYM returned data — consider switching to SPYM if SPLG is empty.")
    else:
        print("SPYM is not returning usable data (empty or error).")

    # Exit code: 0 if at least one ticker returns non-empty, else 2
    non_empty = sum(int(r.get("ok") and not r.get("empty")) for r in results.values())
    if non_empty == 0:
        print("\nNo usable data returned for either SPLG or SPYM.")
        return 2

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
