from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from core.features.feature_base import FeatureBase, FeatureConfig


class TrendFuryConfig(FeatureConfig):
    name: str = "trend_fury"
    window: int = 50  # Rolling window size for regression
    vwap_window: int = 50  # Rolling window size for VWAP
    use_returns: bool = False  # Option to use returns instead of prices
    use_ema: bool = False  # Option to use exponential moving average
    use_volume_weighting: bool = False  # Option to use volume-weighted regression
    volume_normalization_window: int = 50  # Window size for volume normalization
    cum_diff_quantile_threshold: float = 0.5  # Threshold for significant slope changes
    reversal_sensitivity: float = 0.3  # Sensitivity for detecting reversals
    slope_quantile_threshold: float = 0.4  # Threshold for slope quantile
    use_vwap_filter: bool = False  # Toggle for using VWAP based signal filtering
    use_slope_filter: bool = False  # Toggle for using slope based signal filtering


class TrendFury(FeatureBase[TrendFuryConfig]):
    def calculate(self, candles: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the rolling regression slope and generate trading signals.
        """
        # Extract configuration parameters
        window = self.config.window
        vwap_window = self.config.vwap_window
        use_returns = self.config.use_returns
        use_ema = self.config.use_ema
        use_volume_weighting = self.config.use_volume_weighting
        volume_normalization_window = self.config.volume_normalization_window
        cum_diff_quantile_threshold = self.config.cum_diff_quantile_threshold
        reversal_sensitivity = self.config.reversal_sensitivity
        use_vwap_filter = self.config.use_vwap_filter
        use_slope_filter = self.config.use_slope_filter
        slope_quantile_threshold = self.config.slope_quantile_threshold

        # Ensure necessary columns exist
        required_columns = ['close', 'quote_asset_volume', 'taker_buy_quote_volume']
        for col in required_columns:
            if col not in candles.columns:
                raise ValueError(f"Data does not contain '{col}' column required for trend calculation.")

        # Create a copy of the candles to avoid modifying the original DataFrame
        candles = candles.copy()

        # Prepare the price series
        if use_returns:
            # Compute returns (log returns)
            candles['price_series'] = np.log(candles['close'] / candles['close'].shift(1))
        elif use_ema:
            # Use exponential moving average
            candles['price_series'] = candles['close'].ewm(span=window, adjust=False).mean()
        else:
            candles['price_series'] = candles['close']

        # Calculate taker sell quote volume
        candles['taker_sell_quote_volume'] = candles['quote_asset_volume'] - candles['taker_buy_quote_volume']

        # Compute volume weights
        if use_volume_weighting:
            # Use total quote asset volume as weights
            # Normalize the volumes using a rolling window to prevent extreme values
            average_rolling_volume = candles['quote_asset_volume'].rolling(
                volume_normalization_window, min_periods=1
            ).mean()
            candles['volume_weight'] = candles['quote_asset_volume'] / average_rolling_volume
        else:
            candles['volume_weight'] = 1.0  # Equal weighting

        # Calculate rolling regression slopes
        candles['slope'] = candles['price_series'].rolling(
            window=window, min_periods=window
        ).apply(
            lambda x: self.calculate_slope(
                x,
                weights=candles['volume_weight'].loc[x.index] if use_volume_weighting else None
            ),
            raw=False
        )

        # Calculate slope differences (rate of change of the slope)
        candles['slope_diff'] = candles['slope'].diff()
        candles['cumulative_slope_diff'] = self.cumsum_reset_on_reversal(
            candles['slope_diff'], reversal_threshold=reversal_sensitivity)
        # Add additional computed columns
        # Calculate the ratio of taker buy volume to total volume
        candles['taker_buy_volume_ratio'] = candles['taker_buy_quote_volume'] / candles['quote_asset_volume']
        # Calculate the cumulative volume-weighted price
        candles['cum_volume'] = candles['quote_asset_volume'].cumsum()
        candles['cum_volume_price'] = (candles['close'] * candles['quote_asset_volume']).cumsum()
        candles['vwap'] = candles['cum_volume_price'] / candles['cum_volume']
        candles["rolling_cum_volume"] = candles["quote_asset_volume"].rolling(window=vwap_window).sum()
        candles["rolling_cum_volume_price"] = (candles['close'] * candles['quote_asset_volume']).rolling(window=vwap_window).sum()
        candles["rolling_vwap"] = candles["rolling_cum_volume_price"] / candles["rolling_cum_volume"]
        positive_slope_quantile_threshold = candles[candles["slope"] > 0]["slope"].quantile(slope_quantile_threshold)
        negative_slope_quantile_threshold = candles[candles["slope"] < 0]["slope"].quantile(slope_quantile_threshold)
        pos_cum_slope_diff_thresh = candles[
            candles["cumulative_slope_diff"] > 0]["cumulative_slope_diff"].quantile(cum_diff_quantile_threshold)
        neg_cum_slope_diff_thresh = candles[
            candles["cumulative_slope_diff"] < 0]["cumulative_slope_diff"].quantile(1 - cum_diff_quantile_threshold)
        # Conditional VWAP calculation and signal generation
        candles['signal'] = 0
        candles.loc[
            (candles["cumulative_slope_diff"] == 0) &
            (candles['cumulative_slope_diff'] - candles["cumulative_slope_diff"].shift(1) > pos_cum_slope_diff_thresh) &
            (candles['close'] < candles['rolling_vwap'] if use_vwap_filter else True) &
            (candles['slope'] < negative_slope_quantile_threshold if use_slope_filter else True), 'signal'] = 1
        candles.loc[
            (candles["cumulative_slope_diff"] == 0) &
            (candles['cumulative_slope_diff'] - candles["cumulative_slope_diff"].shift(1) < neg_cum_slope_diff_thresh) &
            (candles['close'] > candles['rolling_vwap']) &
            (candles['slope'] > positive_slope_quantile_threshold if use_slope_filter else True), 'signal'] = -1

        return candles

    @staticmethod
    def calculate_slope(values: pd.Series, weights: Optional[pd.Series] = None) -> float:
        """
        Calculate the slope (trend) of a time series using linear regression.
        If weights are provided, perform a weighted regression.
        """
        if len(values) < 2:
            return 0.0

        # Prepare the data for regression
        X = np.arange(len(values)).reshape(-1, 1)
        y = values.values.reshape(-1, 1)

        if weights is not None:
            weights = weights.values.flatten()
            # Fit weighted linear regression
            model = LinearRegression().fit(X, y, sample_weight=weights)
        else:
            # Fit ordinary linear regression
            model = LinearRegression().fit(X, y)

        return model.coef_[0][0]

    @staticmethod
    def cumsum_reset_on_reversal(series, reversal_threshold=0.3):
        """
        Calculate cumulative sum with reset on trend reversal, using a percentage-based threshold and smoothing window.
        :param series: Input series of slope differences
        :param reversal_threshold: Percentage of trend reversal required to reset (default: 0.3 or 30%)
        :param smoothing_window: Window size for moving average smoothing (default: 3)
        :return: Series of cumulative sums with resets on significant reversals
        """
        cumsum = 0
        max_cumsum = 0
        min_cumsum = 0
        output = []
        trend = 0  # 0 for no trend, 1 for uptrend, -1 for downtrend

        for i, change in enumerate(series):
            if np.isnan(change):
                output.append(np.nan)
                continue
            cumsum += series.iloc[i]  # Use original series for cumsum
            if trend == 0:
                # Initialize trend
                if cumsum > 0:
                    trend = 1
                    max_cumsum = cumsum
                elif cumsum < 0:
                    trend = -1
                    min_cumsum = cumsum
            elif trend > 0:
                max_cumsum = max(max_cumsum, cumsum)
                if cumsum <= max_cumsum * (1 - reversal_threshold):
                    # Significant downward reversal
                    cumsum = 0
                    max_cumsum = 0
                    min_cumsum = 0
                    trend = -1
            elif trend < 0:
                min_cumsum = min(min_cumsum, cumsum)
                if cumsum >= min_cumsum * (1 - reversal_threshold):
                    # Significant upward reversal
                    cumsum = 0
                    max_cumsum = 0
                    min_cumsum = 0
                    trend = 1

            output.append(cumsum)

        return pd.Series(output, index=series.index)
