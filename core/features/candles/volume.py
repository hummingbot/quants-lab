import pandas as pd

from core.features.feature_base import FeatureBase, FeatureConfig


class VolumeConfig(FeatureConfig):
    name: str = "volume"
    short_window: int = 5
    long_window: int = 100


class Volume(FeatureBase[VolumeConfig]):
    def calculate(self, candles: pd.DataFrame):
        # Ensure the required columns are present
        required_columns = ['volume', 'close', 'taker_buy_base_volume']
        for col in required_columns:
            if col not in candles.columns:
                raise ValueError(f"Candles DataFrame does not contain '{col}' column required for volume calculation.")

        # Calculate volume metrics
        candles["volume_usd"] = candles["volume"] * candles["close"]
        candles["buy_taker_volume_usd"] = candles["taker_buy_base_volume"] * candles["close"]
        candles["sell_taker_volume_usd"] = candles["volume_usd"] - candles["buy_taker_volume_usd"]

        # Calculate buy/sell imbalance
        candles["buy_sell_imbalance"] = candles["buy_taker_volume_usd"] - candles["sell_taker_volume_usd"]

        # Calculate rolling metrics for short and long windows
        self.add_rolling_metrics(candles, self.config.short_window, "short")
        self.add_rolling_metrics(candles, self.config.long_window, "long")

        return candles

    def add_rolling_metrics(self, df: pd.DataFrame, window: int, suffix: str):
        # Calculate rolling total volume
        rolling_total_volume_usd = df["volume_usd"].rolling(window=window, min_periods=1).sum()

        # Calculate rolling buy/sell imbalance
        df[f"rolling_buy_sell_imbalance_{suffix}"] = df["buy_sell_imbalance"].rolling(
            window=window, min_periods=1).sum() / rolling_total_volume_usd

        # Calculate rolling buy/sell pressure
        df[f"rolling_buy_sell_pressure_{suffix}"] = df["buy_taker_volume_usd"].rolling(
            window=window, min_periods=1).sum() / df["sell_taker_volume_usd"].rolling(
            window=window, min_periods=1).sum()

        # Handle potential division by zero
        df[f"rolling_buy_sell_pressure_{suffix}"] = df[f"rolling_buy_sell_pressure_{suffix}"].replace(
            [float('inf'), -float('inf')], 0)
