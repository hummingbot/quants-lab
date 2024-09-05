import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.features.feature_base import FeatureBase, FeatureConfig


class TrendConfig(FeatureConfig):
    name: str = "trend"
    short_window: int = 50
    long_window: int = 200


class Trend(FeatureBase[TrendConfig]):
    def calculate(self, candles):
        # Extract the configuration parameters
        short_window = self.config.short_window
        long_window = self.config.long_window

        # Ensure the candles has the required 'close' column
        if 'close' not in candles.columns:
            raise ValueError("Data handler does not contain 'close' column required for trend calculation.")

        # Calculate the trend score for each row
        candles = candles.copy()
        trend_score, short_slope, long_slope = self.calculate_trending_score(candles['close'], short_window, long_window)
        candles['trend_score'] = trend_score
        candles['short_slope'] = short_slope
        candles['long_slope'] = long_slope
        return candles

    def calculate_trending_score(self, series: pd.Series, short_window: int, long_window: int):
        short_mavg = series.rolling(window=short_window, min_periods=1).mean()
        long_mavg = series.rolling(window=long_window, min_periods=1).mean()

        short_slope = short_mavg.rolling(window=short_window, min_periods=1).apply(self.calculate_slope, raw=True)
        long_slope = long_mavg.rolling(window=long_window, min_periods=1).apply(self.calculate_slope, raw=True)

        trending_score = (short_slope - long_slope) / (np.abs(short_slope) + np.abs(long_slope))
        return trending_score, short_slope, long_slope

    @staticmethod
    def calculate_slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        X = np.arange(len(values)).reshape(-1, 1)
        y = values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
