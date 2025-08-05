"""
binance-futures.py – Incremental Binance USDⓈ-M klines downloader (int64 timestamps).
"""

from __future__ import annotations
import os
import sys
import math
import time
import argparse
import logging
import datetime as dt
from typing import List, Dict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from binance.um_futures import UMFutures

# ───────── Config ─────────
DATA_DIR         = os.path.join(os.path.dirname(__file__), "binance-futures")
MAX_LIMIT, PAUSE = 1500, 0.05
FALLBACK_START   = 1_577_836_800_000          # 2020-01-01 UTC (ms)
os.makedirs(DATA_DIR, exist_ok=True)

# ───────── Helpers ─────────
def ms_interval(code: str) -> int:
    n, unit = int(code[:-1]), code[-1]
    return n * {"m": 60_000, "h": 3_600_000,
                "d": 86_400_000, "w": 604_800_000}[unit]

def safe_last_ms(path: str) -> int | None:
    """Return last open_time as int-ms, or None. Handles empty/corrupt files."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        if os.path.exists(path):
            os.remove(path)
        return None
    try:
        pf = pq.ParquetFile(path)
        rg  = pf.metadata.num_row_groups - 1
        val = pf.read_row_group(rg, ["open_time"]).column(0)[-1].as_py()
        if isinstance(val, (dt.datetime, pd.Timestamp)):      # ← ensure int-ms
            val = int(val.timestamp() * 1000)
        return val
    except Exception as err:
        os.rename(path, f"{path}.corrupt")
        print(f"⚠️  moved unreadable {path} → {path}.corrupt ({err})",
              file=sys.stderr)
        return None

def write_df(df: pd.DataFrame, path: str):
    """Append df to Parquet; fast on pyarrow<15, safe rewrite on ≥15."""
    if df.empty:
        return
    tbl_new   = pa.Table.from_pandas(df, preserve_index=False)
    append_ok = int(pa.__version__.split(".", 1)[0]) < 15

    if not os.path.exists(path):
        pq.write_table(tbl_new, path, compression="zstd")
        return

    if append_ok:
        pq.write_table(tbl_new, path, compression="zstd", append=True)
    else:
        tbl_all = pa.concat_tables([pq.read_table(path), tbl_new])
        pq.write_table(tbl_all, path, compression="zstd")
        if not write_df.__dict__.get("warned"):
            print(
                f"ℹ️  PyArrow {pa.__version__} lacks fast append; rewriting file."
            )
            write_df.warned = True

# ───────── Binance fetch ─────────
def fetch_chunk(cli: UMFutures, sym: str, intr: str,
                start: int, end: int) -> pd.DataFrame:
    kl = cli.klines(symbol=sym, interval=intr,
                    startTime=start, endTime=end, limit=MAX_LIMIT)
    if not kl:
        return pd.DataFrame()

    cols = ["open_time","open","high","low","close","volume",
            "close_time","quote_vol","trade_count",
            "taker_base_vol","taker_quote_vol","ignore"]

    df = (pd.DataFrame(kl, columns=cols)
            .drop(columns=["ignore", "close_time"])
            .astype({
                "open": float, "high": float, "low": float, "close": float,
                "volume": float, "quote_vol": float,
                "taker_base_vol": float, "taker_quote_vol": float,
                "trade_count": int
            }))
    return df

# ───────── per-symbol loop ─────────
def download_symbol(cli: UMFutures, info: Dict,
                    intr: str, now: int):
    sym   = info["symbol"]
    path  = os.path.join(DATA_DIR, f"{sym}_{intr}.parquet")
    step  = ms_interval(intr)

    last  = safe_last_ms(path)
    start = info.get("onboardDate", FALLBACK_START) if last is None else last + step
    if start >= now:
        print(f"{sym:<12} up-to-date")
        return

    total = math.ceil((now - start) / step)
    start_dt = dt.datetime.fromtimestamp(start/1000, tz=dt.timezone.utc)
    print(f"{sym:<12} start {start_dt:%Y-%m-%d} → now  ({total:,} bars)")

    cur, done = start, 0
    while cur < now:
        try:
            df = fetch_chunk(cli, sym, intr, cur, now)
        except Exception as e:
            print(f"\n{sym} API error: {e} – retry in 3 s …", file=sys.stderr)
            time.sleep(3)
            continue

        if df.empty:
            break
        write_df(df, path)
        done += len(df)
        cur   = df["open_time"].iloc[-1] + step          # open_time is int64
        pct   = done / total
        print(f"\r{sym:<12} {done:>10,}/{total:<10,} ({pct:5.1%})",
              end="", flush=True)
        time.sleep(0.05)
    print()

# ───────── Main ─────────
def get_symbols(cli: UMFutures) -> List[Dict]:
    ei = cli.exchange_info()
    return [s for s in ei["symbols"]
            if s.get("contractType") == "PERPETUAL"
            and s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"]

def main():
    ap = argparse.ArgumentParser(description="Binance USDⓈ-M klines → Parquet")
    ap.add_argument("--interval", default="1m", help="1m 5m 1h … (default 1m)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.ERROR)
    cli, now_ms = UMFutures(), int(time.time()*1000)

    try:
        for info in get_symbols(cli):
            download_symbol(cli, info, args.interval, now_ms)
    except KeyboardInterrupt:
        print("\nInterrupted – partial data safely stored.")

if __name__ == "__main__":
    main()
