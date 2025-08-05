import numpy as np
from nexus.indicators.indicators import SMA, sma_cython


def test_sma_cython_matches_python():
    data = np.random.random(100).astype(float)
    window = 10
    py_val = SMA(data, window)
    cy_val = sma_cython(data, window)
    assert abs(py_val - cy_val) < 1e-9
