"""
Ehlers' Super Smoother, Ultimate Smoother, Channel and Bands
===========================================
A pure‑Python implementation of the indicators described by John Ehlers in
TASC 3/24 and 4/24 and ported to Lite‑C on *The Financial Hacker* website.

The module exposes three classes — `UltimateSmoother`, `UltimateChannel`,
`UltimateBands` — each with an incremental `update()` method so they can be
used either in vectorised form or bar‑by‑bar inside a trading loop.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from nexus.indicators.utils import push

"""
// Ehlers' smoothing filter ("SuperSmoother"), 2-pole Butterworth * SMA
var Smooth(var *Data,int Cutoff)
{
	if(!Cutoff) Cutoff = 10;
	var f = (1.414*PI) / Cutoff;
	var a = exp(-f);
	var c2 = 2*a*cos(f);
	var c3 = -a*a;
	var c1 = 1 - c2 - c3;

	var* Filt = series(*Data,4);
	SETSERIES(Data,0);
	return Filt[0] = c1*(Data[0]+Data[1])*0.5 + c2*Filt[1] + c3*Filt[2];
}
"""


class SuperSmoother:
    """
    Ehlers' smoothing filter ("SuperSmoother"), 2-pole Butterworth * SMA
    Parameters
    ----------
    data : 1-D array-like, chronological (oldest → newest)
        The raw price/indicator series to be smoothed.
    cutoff : int, default 10
        Cut-off period in bars. The shorter the period, the faster
        the filter responds.

    """

    def __init__(self, cutoff: int = 10):
        self.f = (1.414 * math.pi) / cutoff
        self.a = math.exp(-self.f)
        self.c2 = 2 * self.a * math.cos(self.f)
        self.c3 = -self.a * self.a
        self.c1 = 1 - self.c2 - self.c3
        self.arr = np.zeros(4)
        self.initialized = False

    def update(self, data: float) -> float:
        if not self.initialized:
            self.arr[:] = data
            self.initialized = True
            return data

        self.arr[1:] = self.arr[:-1]  # Shifts elements to the right
        self.arr[0] = (
            self.c1 * (data) * 1 + self.c2 * self.arr[1] + self.c3 * self.arr[2]
        )
        #     ToDo          update formula! data can be array
        return self.arr[0]


#
#  Ehlers' Ultimate Smoother (2‑pole IIR)
#
class UltimateSmoother:
    """Incremental 2-pole low-lag smoother (John Ehlers, 2024).

    Parameters
    ----------
    length : int
        Smoothing length (*L* in the article). A typical value is **20**.
    """

    def __init__(self, length: int = 20):
        if length <= 0:
            raise ValueError("length must be positive")

        f = (1.414 * math.pi) / length
        a1 = math.exp(-f)
        self.c2 = 2.0 * a1 * math.cos(f)
        self.c3 = -a1 * a1
        self.c1 = (1.0 + self.c2 - self.c3) / 4.0
        self.data = np.zeros(4)
        self.us = np.zeros(4)
        self.initialized = False

    def update(self, data: float) -> float:
        """Feed a new price, return the latest smoothed value."""
        push(self.data, data)  # insert new value at index [0]

        if not self.initialized:
            # bootstrap: replicate price until we have enough history
            self.us[:] = data
            self.data[:] = data
            self.initialized = True

        self.us[1:] = self.us[:-1]  # Shifts elements to the right
        # current smoother output  (see article) — depends on previous outputs
        self.us[0] = (
            (1.0 - self.c1) * self.data[0]
            + (2.0 * self.c1 - self.c2) * self.data[1]
            - (self.c1 + self.c3) * self.data[2]
            + self.c2 * self.us[0]
            + self.c3 * self.us[1]
        )

        return self.us[0]


#
# Ultimate Channel
#
class UltimateChannel:
    """True‑Range based dynamic channel (Ehlers 2024).

    When price touches `upper` or `lower`, you may interpret it as an
    overbought / oversold condition for mean‑reversion or breakout trades.

    Parameters
    ----------
    length : int
        Smoother length for the *Center* line.
    str_length : int
        Smoother length for the *True‑Range* ("STR") component.
    num_strs : float
        Multiplier for the channel half‑width (number of "STRs").
    """

    def __init__(self, length: int = 20, str_length: int = 20, num_strs: float = 1.0):
        self.center_sm = UltimateSmoother(length)
        self.str_sm = UltimateSmoother(str_length)
        self.k = num_strs
        self.prev_close: float | None = None

    # ──────────────────────────────────────────────────────────
    def update(
        self, close: float, high: float, low: float
    ) -> Tuple[float, float, float]:
        """Return *(center, upper, lower)* for the current bar."""
        if self.prev_close is None:
            self.prev_close = float(close)  # seed previous‑close

        th = max(self.prev_close, high)  # true high
        tl = min(self.prev_close, low)  # true low
        str_val = self.str_sm.update(th - tl)  # smoothed true range
        center = self.center_sm.update(close)
        upper = center + self.k * str_val
        lower = center - self.k * str_val
        self.prev_close = float(close)
        return center, upper, lower


#
#  Ultimate Bands
#
class UltimateBands:
    """Standard‑deviation envelope around the Ultimate Smoother.

    Parameters
    ----------
    length : int
        Look‑back window for both the smoother and the rolling variance.
    num_sds : float
        Band width in population standard deviations.
    """

    def __init__(self, length: int = 20, num_sds: float = 1.0):
        self.center_sm = UltimateSmoother(length)
        self.L = length
        self.k = num_sds
        # ring buffer for *diff* (price − center) — fixed length *L*
        self._buf: List[float] = []

    # ──────────────────────────────────────────────────────────
    def update(self, close: float) -> Tuple[float, float, float]:
        center = self.center_sm.update(close)
        diff = close - center
        self._buf.append(diff)
        if len(self._buf) > self.L:
            self._buf.pop(0)

        if len(self._buf) < self.L:
            # insufficient history → bands undefined
            return center, math.nan, math.nan

        diffs = np.asarray(self._buf, dtype=float)
        sd = math.sqrt((diffs**2).sum() / self.L)  # population SD
        upper = center + self.k * sd
        lower = center - self.k * sd
        return center, upper, lower
