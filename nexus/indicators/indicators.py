import numpy as np
from nexus.indicators.utils import push

try:
    from ._cython_indicators import sma_cython
except Exception:
    # Compile on the fly if Cython extension is missing
    try:
        import pyximport
        import numpy
        pyximport.install(language_level=3, setup_args={'include_dirs': numpy.get_include()})
        from ._cython_indicators import sma_cython  # type: ignore
    except Exception:  # pragma: no cover - fall back to pure Python
        sma_cython = None

def SMA(time_series, window):
    """
    Calculates the moving average of the given series.

    Parameters:
    - series (numpy array or list): The timeseries of values. 
    - window (int): The window size for the moving average.

    Returns:
    - float: The moving average value.
    """
    # ``time_series`` can be either a list or a ``numpy.ndarray``.  The
    # previous implementation attempted to access the ``data`` attribute
    # of the array which returns a buffer object rather than the array
    # values.  Summing over that buffer results in incorrect averages.
    # Instead, operate directly on the sequence itself.

    # Convert to ``numpy`` array for slice operations and to ensure
    # ``len`` works consistently with lists.
    arr = np.asarray(time_series, dtype=float)

    if sma_cython is not None:
        return float(sma_cython(arr, int(window)))

    if len(arr) < window:
        return float(np.sum(arr[:len(arr)]) / len(arr))
    else:
        return float(np.sum(arr[:window]) / window)

class ATR:
    """
    Class to compute the Average True Range (ATR) indicator.
    """

    def __init__(self, period):
        """
        Initialize the ATR indicator.

        Parameters:
        - period (int): The time period over which to calculate ATR.
        """
        self.period = max(1, period)
        self.atr_series = np.zeros(2)
        self.prev_close = None  # To store the previous close price

    def update(self, high, low, close):
        """
        Update the ATR with new high, low, and close prices.

        Parameters:
        - high (float): The current high price.
        - low (float): The current low price.
        - close (float): The current close price.

        Returns:
        - float: The current ATR value.
        """
        # For the first data point, we cannot compute ATR
        if self.prev_close is None:
            # Initialize ATR with high - low of the first bar
            tr = high - low
            atr = tr  # Initial ATR value
        else:
            # Calculate True Range (TR)
            tr = max(high, self.prev_close) - min(low, self.prev_close)
            # Retrieve the previous ATR value
            prev_atr = self.atr_series[0] if len(self.atr_series) > 0 else tr
            # Calculate current ATR
            atr = ((prev_atr * (self.period - 1)) + tr) / self.period

        # Update the ATR series
        push(self.atr_series, atr)

        # Update the previous close price
        self.prev_close = close

        return atr

    def value(self):
        """
        Get the latest ATR value.

        Returns:
        - float: The latest ATR value.
        """
        if len(self.atr_series) > 0:
            return self.atr_series[0]
        else:
            return None

def aci(data: np.ndarray, period: int, lag: int) -> float:
    """
    Auto-Correlation Index (ACI)

    Parameters
    ----------
    data : 1-D array-like
        Price series in chronological order (oldest first).
        Must contain at least `period` elements.
    period : int
        Total look-back window.
    lag : int
        Maximum lag to average.  Will be clipped to ``1 … period-1``.

    Returns
    -------
    float
        The average Pearson autocorrelation for lags 1 … `lag`
        over the last `period-lag` bars.
    """
    period = int(period)
    lag         = int(np.clip(lag, 1, period - 1))
    win_len     = period - lag
    a           = data[-win_len:]                 # latest window
    corr_sum    = 0.0
    for k in range(1, lag + 1):
        b = data[-win_len-k:-k]                  # shift by +k
        corr_sum += np.corrcoef(a, b)[0, 1]
    return corr_sum / lag

def fractal_dimension(data: np.ndarray, period: int):
    """
    Peters-style Fractal Dimension

    Parameters
    ----------
    data : 1-D array-like
        Price series in chronological order (oldest first).
        Must contain at least `period` elements.
    period : int
        Look-back window in bars. Will be forced to an even number,
        exactly like `period &= ~1` in the Lite-C source.

    Returns
    -------
    float
        Fractal dimension in the range ~[1,2].
        • ≈1   → very smooth / trending  
        • ≈2   → highly jagged / random walk
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError("data must be 1-D")

    # --- sanitise the look-back length ------------------------------------
    period = int(period) & ~1           # force even
    if period < 2 or len(x) < period:
        raise ValueError("`period` must be ≥2 and ≤ len(data)`")

    p2   = max(1, period // 2)
    win  = x[-period:]                  # last `period` samples
    seg1 = win[-p2:]                    # most-recent half
    seg2 = win[:p2]                     # earlier half

    n1 = (seg1.max() - seg1.min()) / p2
    n2 = (seg2.max() - seg2.min()) / p2
    n3 = (win .max() - win .min()) / period

    if (n1 + n2) <= 0.0 or n3 <= 0.0:   # guard against zeros
        return 1.0

    return (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)

def hurst_exponent(data: np.ndarray, period: int):
    """
    Hurst exponent.

    Parameters
    ----------
    data : 1-D array-like
        Price series in chronological order (oldest first).
        Must contain at least `period` elements.
    period : int
        Look-back window. Values < 20 are forced up to 20, exactly
        like `Period = Max(20, Period)` in the script.

    Returns
    -------
    float
        Hurst exponent in the range [0, 1].
        • ≈0 → anti-persistence / mean-reversion  
        • ≈0.5 → random walk  
        • ≈1 → strong persistence / trend
    """
    period = max(20, period)
    h_raw  = 2.0 - fractal_dimension(data, period)  

    return float(np.clip(h_raw, 0.0, 1.0))

def mmi(data: np.ndarray, period: int):
    """
    Market Meanness Index.

    Parameters
    ----------
    data : 1-D array-like
        Price series in chronological order (oldest first).
        Must contain at least `period` elements.
    period : int
        Look-back window (≥ 2).

    Returns
    -------
    float
        MMI in percent (0 … 100).  
        • high  → choppy / mean-reverting market  
        • low   → persistent trend
    """
    period = int(period)
    if period < 2:
        raise ValueError("`period` must be ≥ 2")
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError("`data` must be 1-D")
    if len(x) < period:
        raise ValueError("Need at least `period` samples")

    win  = x[-period:]          # last `period` bars, chronological
    med  = np.median(win)
    rev  = win[::-1]            # newest bar at index 0 (Zorro layout)

    # vectorised replica of the Lite-C loop
    up   = (rev[1:] > med) & (rev[1:] > rev[:-1])
    dn   = (rev[1:] < med) & (rev[1:] < rev[:-1])

    return 100.0 * (up.sum() + dn.sum()) / (period - 1)
