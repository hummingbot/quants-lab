import numpy as np
from timeit import timeit
from nexus.indicators.indicators import SMA
try:
    from nexus.indicators._cython_indicators import sma_cython
except Exception:
    sma_cython = None

data = np.random.random(1000000)


def bench_py():
    SMA(data, 5)


def bench_cy():
    if sma_cython is None:
        return
    sma_cython(data, 5)

if __name__ == "__main__":
    py_time = timeit(bench_py, number=1000)
    print(f"Pure Python SMA: {py_time:.4f}s")
    if sma_cython is not None:
        cy_time = timeit(bench_cy, number=1000)
        print(f"Cython SMA: {cy_time:.4f}s")
        print(f"Speedup: {py_time / cy_time:.2f}x")
    else:
        print("Cython version not available")
