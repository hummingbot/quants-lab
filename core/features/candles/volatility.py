import pandas_ta as ta  # noqa: F401

from core.features.feature_base import FeatureBase, FeatureConfig


class VolatilityConfig(FeatureConfig):
    name: str = "volatility"
    window: int = 100


class Volatility(FeatureBase[VolatilityConfig]):
    def calculate(self, candles):
        window = self.config.window
        candles["volatility"] = candles["close"].pct_change().rolling(window=window).std()
        candles["natr"] = ta.natr(candles["high"], candles["low"], candles["close"], length=window) / 100
        bbands = ta.bbands(candles["close"], length=window)
        candles["bb_width"] = bbands[f"BBB_{window}_2.0"]
        return candles
