#!/usr/bin/env python3
"""
plot.py — Fast candlestick viewer for large klines files (CSV or Parquet).

New flags:
    --start   YYYY-MM-DD   : first timestamp (UTC) to plot
    --end     YYYY-MM-DD   : last  timestamp (UTC) to plot
    --last    N            : keep only the last N rows
    --resample 5T|15T|1H   : aggregate to a coarser bar
    --thin    N            : max number of candles after thinning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from nexus.charting import show_plotly_figure

_PARQUET = {".parquet", ".parq", ".pq"}


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix.lower() in _PARQUET else pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


def _slice(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.start or args.end:
        m = pd.Series(True, index=df.index)
        if args.start:
            m &= df["open_time"] >= pd.Timestamp(args.start, tz="UTC")
        if args.end:
            m &= df["open_time"] <= pd.Timestamp(args.end, tz="UTC")
        df = df[m]
    if args.last:
        df = df.tail(args.last)
    return df.reset_index(drop=True)


def _resample_or_thin(df: pd.DataFrame, args) -> pd.DataFrame:
    if args.resample:
        args.resample = args.resample.replace("T", "min")
        df = (
            df.set_index("open_time")
            .resample(args.resample)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )
    if args.thin and len(df) > args.thin:
        step = len(df) // args.thin
        df = df.iloc[::step]
    return df


def _figure(df: pd.DataFrame, title: str | None) -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=df["open_time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color="green",
            decreasing_line_color="red",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (UTC)",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
    )
    return fig


def main():
    p = argparse.ArgumentParser(
        description="Fast candlestick chart for large klines files."
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--title")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--last", type=int)
    p.add_argument(
        "--resample",
        help="Pandas offset alias like 5T, 15T, 1H (takes precedence over --thin)",
    )
    p.add_argument(
        "--thin", type=int, help="Maximum number of candles to display after thinning"
    )
    args = p.parse_args()

    path = Path(args.input)
    path.exists() or sys.exit(f"No such file: {path}")
    df = _read(path)
    df = _slice(df, args)
    df = _resample_or_thin(df, args)

    fig = _figure(df, args.title)
    if args.output:
        fig.write_html(args.output)
        print(f"✓ saved to {args.output}")
    else:
        show_plotly_figure(fig)  # WSL workaround, instead of fig.show()


if __name__ == "__main__":
    main()
