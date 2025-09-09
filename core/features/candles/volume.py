import pandas as pd
import numpy as np

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

        # Calculate base volume metrics
        candles["volume_usd"] = candles["volume"] * candles["close"]
        candles["buy_volume_usd"] = candles["taker_buy_base_volume"] * candles["close"]
        candles["sell_volume_usd"] = candles["volume_usd"] - candles["buy_volume_usd"]

        # Calculate buy percentage (0 to 1, where 0.5 is neutral)
        candles["buy_percentage"] = candles["buy_volume_usd"] / candles["volume_usd"]
        candles["buy_percentage"] = candles["buy_percentage"].fillna(0.5)
        
        # Simple buy/sell imbalance (-1 to 1, where 0 is neutral)
        candles["buy_sell_imbalance"] = (candles["buy_percentage"] - 0.5) * 2
        
        # Short-term buy pressure (smoothed)
        candles["buy_pressure_short"] = candles["buy_percentage"].rolling(
            window=self.config.short_window, min_periods=1
        ).mean()
        
        # Long-term buy pressure (smoothed)
        candles["buy_pressure_long"] = candles["buy_percentage"].rolling(
            window=self.config.long_window, min_periods=1
        ).mean()
        
        # Buy pressure divergence: short vs long
        # Positive = bullish (short-term buying > long-term)
        # Negative = bearish (short-term selling > long-term)
        candles["buy_pressure_divergence"] = candles["buy_pressure_short"] - candles["buy_pressure_long"]
        
        # Buy momentum (rate of change)
        candles["buy_momentum"] = candles["buy_pressure_short"].diff()
        
        # Volume moving averages
        candles["volume_ma_short"] = candles["volume_usd"].rolling(
            window=self.config.short_window, min_periods=1
        ).mean()
        candles["volume_ma_long"] = candles["volume_usd"].rolling(
            window=self.config.long_window, min_periods=1
        ).mean()
        
        # Volume surge (current vs long-term average)
        candles["volume_surge"] = candles["volume_usd"] / candles["volume_ma_long"]
        candles["volume_surge"] = candles["volume_surge"].replace([np.inf, -np.inf], 1).fillna(1)
        
        # Combined buy signal: volume surge * buy pressure divergence
        candles["volume_buy_signal"] = candles["volume_surge"] * candles["buy_pressure_divergence"]
        
        # Trend consistency (lower std = more consistent)
        candles["trend_consistency"] = 1 - candles["buy_percentage"].rolling(
            window=self.config.short_window, min_periods=1
        ).std()
        candles["trend_consistency"] = candles["trend_consistency"].fillna(0.5)
        
        # Market regime scores
        # Accumulation: sustained buying with high volume
        candles["accumulation_score"] = (
            candles["buy_pressure_long"] * candles["volume_surge"] * candles["trend_consistency"]
        )
        # Distribution: sustained selling with high volume
        candles["distribution_score"] = (
            (1 - candles["buy_pressure_long"]) * candles["volume_surge"] * candles["trend_consistency"]
        )
        
        return candles
