import numpy as np
import polars as pl
from time import perf_counter

N = 100_000       # number of push iterations
LOOK_BACK = 100   # length of the series

def push_np(arr, value):
    """Shift elements in-place so index 0 holds 'value'."""
    arr[1:] = arr[:-1]
    arr[0] = value

def push_pl(series, value):
    """Prepend 'value' so index 0 is newest."""
    return pl.concat([pl.Series([value]), series.slice(0, len(series) - 1)])

def bench_numpy():
    arr = np.zeros(LOOK_BACK)
    start = perf_counter()
    for i in range(N):
        push_np(arr, float(i))
    return perf_counter() - start

def bench_polars():
    series = pl.Series([0.0] * LOOK_BACK)
    start = perf_counter()
    for i in range(N):
        series = push_pl(series, float(i))
    return perf_counter() - start

if __name__ == "__main__":
    t_np = bench_numpy()
    t_pl = bench_polars()
    print(f"NumPy push:  {t_np:.6f} seconds for {N} iterations")
    print(f"Polars push: {t_pl:.6f} seconds for {N} iterations")
