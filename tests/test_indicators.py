# tests/test_indicators.py

import numpy as np
from nexus.indicators.indicators import SMA, ATR

class TestMovingAverage:
    def test_moving_average_exact_window(self):
        """
        Test moving_average when the series length is exactly the window size.
        """
        data = np.array([10, 20, 30, 40, 50])
        expected_ma = np.sum(data) / 5
        assert SMA(data, window=5) == expected_ma, "Incorrect MA with exact window data."

    def test_moving_average_more_than_window(self):
        """
        Test moving_average when the series length is greater than the window size.
        """
        data = np.array([10, 20, 30, 40, 50, 60, 70])
        expected_ma = (10 + 20 + 30 + 40 + 50) / 5  # 150 / 5 = 30.0
        assert SMA(data, window=5) == expected_ma, "Incorrect MA with more than window data."

    def test_moving_average_static_series(self):
        """
        Test moving_average with a static series.
        """
        data = np.array([10, 20, 30, 40, 50])
        expected_ma = sum(data) / 5
        assert SMA(data, window=5) == expected_ma, "Incorrect MA with static series."

    def test_moving_average_constant_series(self):
        """
        Test moving_average with a constant series.
        """
        data = np.array([10, 10, 10, 10, 10])
        expected_ma = 10.0
        assert SMA(data, window=5) == expected_ma, "Incorrect MA with constant series."

class TestATR:
    def test_atr_initial_value(self):
        """
        Test ATR calculation with the initial data point.
        """
        atr_period = 14
        atr_indicator = ATR(period=atr_period)
        high = 50.0
        low = 45.0
        close = 48.0
        atr_value = atr_indicator.update(high, low, close)
        expected_tr = high - low  # 50 - 45 = 5
        expected_atr = expected_tr  # Initial ATR is TR
        assert atr_value == expected_atr, "Incorrect ATR initial value."

    def test_atr_with_exact_period(self):
        """
        Test ATR calculation when the number of data points equals the period.
        """
        atr_period = 3
        atr_indicator = ATR(period=atr_period)
        # Sample data with known expected ATR values
        data = [
            {'high': 48.70, 'low': 47.79, 'close': 48.16},
            {'high': 48.72, 'low': 48.14, 'close': 48.61},
            {'high': 48.90, 'low': 48.39, 'close': 48.75},
        ]
        expected_atr_values = [0.91, 0.8, 0.7033333333333334]
        for i, bar in enumerate(data):
            atr_value = atr_indicator.update(bar['high'], bar['low'], bar['close'])
            expected_atr = expected_atr_values[i]
            assert abs(atr_value - expected_atr) < 1e-6, \
                f"Incorrect ATR value at index {i}. Expected {expected_atr}, got {atr_value}."

    def test_atr_with_constant_prices(self):
        """
        Test ATR calculation with constant high, low, and close prices.
        """
        atr_period = 14
        atr_indicator = ATR(period=atr_period)
        high = 50.0
        low = 50.0
        close = 50.0
        expected_atr = 0.0
        for _ in range(20):
            atr_value = atr_indicator.update(high, low, close)
            assert atr_value == expected_atr, "ATR should be zero with constant prices."

    def test_atr_with_increasing_volatility(self):
        """
        Test ATR calculation with increasing volatility.
        """
        atr_period = 14
        atr_indicator = ATR(period=atr_period)
        prev_atr = None
        for i in range(1, 21):
            high = 50 + i
            low = 50 - i
            close = 50
            atr_value = atr_indicator.update(high, low, close)
            if prev_atr is not None:
                assert atr_value >= prev_atr, \
                    f"ATR did not increase at index {i}."
            prev_atr = atr_value
