import numpy as np
import pandas as pd
import pandas_ta as ta

from core.features.feature_base import FeatureBase, FeatureConfig


class MeanReversionChannelConfig(FeatureConfig):
    name: str = "mean_reversion_channel"
    length: int = 200
    inner_mult: float = 1.0
    outer_mult: float = 2.415
    source: str = "hlc3"
    filter_type: str = "SuperSmoother"


class MeanReversionChannel(FeatureBase[MeanReversionChannelConfig]):
    def calculate(self, candles: pd.DataFrame) -> pd.DataFrame:
        length = self.config.length
        inner_mult = self.config.inner_mult
        outer_mult = self.config.outer_mult
        source = self.config.source
        filter_type = self.config.filter_type

        # Calculate source
        if source == "hlc3":
            candles["source"] = (candles["high"] + candles["low"] + candles["close"]) / 3
        else:
            candles["source"] = candles[source]

        # Calculate mean line
        if filter_type == "SuperSmoother":
            candles["meanline"] = self.supersmoother(candles["source"], length)
        else:
            candles["meanline"] = self.sak_smoothing(candles["source"], length, filter_type)

        # Calculate mean range
        candles["tr"] = ta.true_range(candles["high"], candles["low"], candles["close"])
        candles["meanrange"] = self.supersmoother(candles["tr"].dropna(), length)

        # Calculate bands
        candles["upband1"] = candles["meanline"] + (candles["meanrange"] * inner_mult * np.pi)
        candles["loband1"] = candles["meanline"] - (candles["meanrange"] * inner_mult * np.pi)
        candles["upband2"] = candles["meanline"] + (candles["meanrange"] * outer_mult * np.pi)
        candles["loband2"] = candles["meanline"] - (candles["meanrange"] * outer_mult * np.pi)

        # Calculate condition
        candles["condition"] = self.calculate_condition(candles)

        return candles

    def supersmoother(self, src, length):
        a1 = np.exp(-np.sqrt(2) * np.pi / length)
        b1 = 2 * a1 * np.cos(np.sqrt(2) * np.pi / length)
        c3 = -a1 * a1
        c2 = b1
        c1 = 1 - c2 - c3

        def smooth(x):
            ss = pd.Series(index=x.index, dtype=float)
            ss.iloc[0] = x.iloc[0]
            ss.iloc[1] = x.iloc[1]
            for i in range(2, len(x)):
                ss.iloc[i] = c1 * x.iloc[i] + c2 * ss.iloc[i-1] + c3 * ss.iloc[i-2]
            return ss

        return src.to_frame().apply(smooth).squeeze()

    def sak_smoothing(self, src, length, filter_type):
        import numpy as np

        c0 = 1.0
        c1 = 0.0
        b0 = 1.0
        b1 = 0.0
        b2 = 0.0
        a1 = 0.0
        a2 = 0.0
        alpha = 0.0
        beta = 0.0
        gamma = 0.0
        cycle = 2 * np.pi / length

        if filter_type == "Ehlers EMA":
            alpha = (np.cos(cycle) + np.sin(cycle) - 1) / np.cos(cycle)
            b0 = alpha
            a1 = 1 - alpha
        elif filter_type == "Gaussian":
            beta = 2.415 * (1 - np.cos(cycle))
            alpha = -beta + np.sqrt((beta * beta) + (2 * beta))
            c0 = alpha * alpha
            a1 = 2 * (1 - alpha)
            a2 = -(1 - alpha) * (1 - alpha)
        elif filter_type == "Butterworth":
            beta = 2.415 * (1 - np.cos(cycle))
            alpha = -beta + np.sqrt((beta * beta) + (2 * beta))
            c0 = alpha * alpha / 4
            b1 = 2
            b2 = 1
            a1 = 2 * (1 - alpha)
            a2 = -(1 - alpha) * (1 - alpha)
        elif filter_type == "BandStop":
            beta = np.cos(cycle)
            gamma = 1 / np.cos(cycle * 2 * 0.1)  # delta default to 0.1
            alpha = gamma - np.sqrt((gamma * gamma) - 1)
            c0 = (1 + alpha) / 2
            b1 = -2 * beta
            b2 = 1
            a1 = beta * (1 + alpha)
            a2 = -alpha
        elif filter_type == "SMA":
            c1 = 1 / length
            b0 = 1 / length
            a1 = 1
        elif filter_type == "EMA":
            alpha = 2 / (length + 1)
            b0 = alpha
            a1 = 1 - alpha
        elif filter_type == "RMA":
            alpha = 1 / length
            b0 = alpha
            a1 = 1 - alpha

        def smooth(x):
            input_1 = x.shift(1).fillna(x)
            input_2 = x.shift(2).fillna(x)
            output_1 = x.shift(1).fillna(0)
            output_2 = x.shift(2).fillna(0)
            input_length = x.shift(length).fillna(x)
            return (c0 * ((b0 * x) + (b1 * input_1) + (b2 * input_2))) + (a1 * output_1) + (a2 * output_2) - (c1 * input_length)

        return src.rolling(window=length).apply(smooth)

    def calculate_condition(self, df):
        conditions = []
        for _, row in df.iterrows():
            if row["close"] > row["meanline"]:
                upband2_1 = row["upband2"] + (row["meanrange"] * 0.5 * 4)
                upband2_9 = row["upband2"] + (row["meanrange"] * 0.5 * -4)
                if row["high"] >= upband2_9 and row["high"] < row["upband2"]:
                    conditions.append(1)
                elif row["high"] >= row["upband2"] and row["high"] < upband2_1:
                    conditions.append(2)
                elif row["high"] >= upband2_1:
                    conditions.append(3)
                elif row["close"] <= row["meanline"] + row["meanrange"]:
                    conditions.append(4)
                else:
                    conditions.append(5)
            elif row["close"] < row["meanline"]:
                loband2_1 = row["loband2"] - (row["meanrange"] * 0.5 * 4)
                loband2_9 = row["loband2"] - (row["meanrange"] * 0.5 * -4)
                if row["low"] <= loband2_9 and row["low"] > row["loband2"]:
                    conditions.append(-1)
                elif row["low"] <= row["loband2"] and row["low"] > loband2_1:
                    conditions.append(-2)
                elif row["low"] <= loband2_1:
                    conditions.append(-3)
                elif row["close"] >= row["meanline"] + row["meanrange"]:
                    conditions.append(-4)
                else:
                    conditions.append(-5)
            else:
                conditions.append(0)
        return pd.Series(conditions, index=df.index)
