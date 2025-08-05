"""python
Ehlers' Ultimate Smoother, Channel and Bands
===========================================
A pure‑Python implementation of the indicators described by John Ehlers in
TASC 3/24 and 4/24 and ported to Lite‑C on *The Financial Hacker* website.

A minimal demo at the bottom loads **BTCUSDT** minute‑klines from the local
folder, applies the indicators, and opens an interactive Plotly chart.
"""
from __future__ import annotations
from pathlib import Path
import plotly.graph_objects as go
from nexus.charting import show_plotly_figure

from nexus.feed.klines import read_klines
from nexus.indicators.ehlers import UltimateChannel, UltimateBands

def demo(symbol: str = "BTCUSDT", data_dir: Path | str = "data", n_rows: int = 1000):
    df = read_klines(Path(data_dir) / f"{symbol}_1m.parquet")
    if n_rows:
        df = df.head(n_rows).copy()

    # instantiate indicators
    chan = UltimateChannel(length=20, str_length=20, num_strs=1)
    band = UltimateBands(length=20, num_sds=1)

    centers, up_ch, lo_ch, up_b, lo_b = [], [], [], [], []
    for _, row in df.iterrows():
        c, u, lo = chan.update(row.close, row.high, row.low)
        centers.append(c)
        up_ch.append(u)
        lo_ch.append(lo)
        c2, u2, l2 = band.update(row.close)
        up_b.append(u2)
        lo_b.append(l2)

    df["center"]        = centers
    df["chan_upper"]    = up_ch
    df["chan_lower"]    = lo_ch
    df["band_upper"]    = up_b
    df["band_lower"]    = lo_b

    # ── Plotly figure ─────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(name="Close",  x=df.open_time, y=df.close,
                             mode="lines", line=dict(width=1)))
    fig.add_trace(go.Scatter(name="Center", x=df.open_time, y=df.center,
                             mode="lines"))

    # Ultimate Channel bands
    fig.add_trace(go.Scatter(name="UC upper", x=df.open_time, y=df.chan_upper,
                             mode="lines"))
    fig.add_trace(go.Scatter(name="UC lower", x=df.open_time, y=df.chan_lower,
                             mode="lines", fill=None))

    # Ultimate Bands
    fig.add_trace(go.Scatter(name="UB upper", x=df.open_time, y=df.band_upper,
                             mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(name="UB lower", x=df.open_time, y=df.band_lower,
                             mode="lines", line=dict(dash="dot")))

    fig.update_layout(title=f"{symbol}: Ultimate Smoother & Bands",
                      xaxis_title="Time", yaxis_title="Price (USDT)")
    
    show_plotly_figure(fig) # WSL workaround, instead of fig.show()


# ╭────────────────────────────────────────────────────────────────╮
# │ 5.  Quick CLI entry‑point                                     │
# ╰────────────────────────────────────────────────────────────────╯
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("Ultimate Smoother demo")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--data-dir", default="history/binance-futures")
    p.add_argument("--rows", default=1000, type=int,
                   help="number of initial rows to plot (0 = all)")
    args = p.parse_args()
    demo(args.symbol, args.data_dir, args.rows)
