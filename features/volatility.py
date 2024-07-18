import pandas_ta as ta  # noqa: F401

from features.feature_base import FeatureBase


class Volatility(FeatureBase):
    def calculate(self, data_handler):
        candles = data_handler.data
        window = self.params.get("window", 100)
        candles["volatility"] = candles["close"].pct_change().rolling(window=window).std()
        candles["natr"] = ta.natr(candles["high"], candles["low"], candles["close"], length=window)
        bbands = ta.bbands(candles["close"], length=window)
        candles["bb_width"] = bbands[f"BBB_{window}_2.0"]
        return candles
