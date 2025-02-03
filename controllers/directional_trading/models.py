import numpy as np


def zscore(df, ma, threshold, col):
    # zscore
    rolling = df[f"{col}"].rolling(ma)
    df["ma"] = rolling.mean()
    df["sd"] = rolling.std()
    df["zscore"] = (df[f"{col}"] - df["ma"]) / df["sd"]
    df["position"] = np.where(df["zscore"] > threshold, 1, 0)
    return df


def rateofchange(df, ma, threshold, col):
    df["%change"] = df[f"{col}"].pct_change()
    df["position"] = np.where(df["%change"] > threshold, 1, 0)
    return df


def robust(df, ma, threshold, col):
    rolling = df[f"{col}"].rolling(ma)
    df["q1"] = rolling.quantile(0.25)
    df["q3"] = rolling.quantile(0.75)
    df["iqr"] = df["q3"] - df["q1"]
    df["median"] = df[f"{col}"].rolling(ma).median()
    df["robust"] = df[f"{col}"] - df["median"] / df["iqr"]
    df["position"] = np.where(df["robust"] > threshold, 1, 0)
    return df


def mabounding(df, ma, threshold, col):
    # mabounding
    rolling = df[f"{col}"].rolling(ma)
    df["ma"] = rolling.mean()
    df["position"] = np.where(df[f"{col}"] > df["ma"] * (1 + threshold), 1, 0)
    return df


def rsi(df, ma, threshold, col):
    delta = df[f"{col}"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(ma).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(ma).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["position"] = np.where(df["rsi"] > threshold, 1, 0)
    return df


def Percentile_MinMaxScaling(df, ma, threshold, col):
    rolling = df[f"{col}"].rolling(ma)
    df["q1"] = rolling.quantile(0.25)
    df["q3"] = rolling.quantile(0.75)
    df["CombineModel"] = 2 * (df[f"{col}"] - df["q1"]) / (df["q3"] - df["q1"]) - 1
    df["position"] = np.where(df["CombineModel"] > threshold, 1, 0)
    return df


def outlier(df, ma, threshold, col):
    rolling = df[f"{col}"].rolling(ma)
    df["q1"] = rolling.quantile(0.25)
    df["q3"] = rolling.quantile(0.75)
    df["iqr"] = df["q3"] - df["q1"]
    df["lowerbound"] = df["q1"] - 3 * df["iqr"]
    df["upperbound"] = df["q3"] + 3 * df["iqr"]
    df["position"] = np.where(df[f"{col}"] > df["upperbound"], 1, 0)
    return df


def stochastic_oscillator(df, ma, threshold, col):
    df["Lowest Low"] = df["low"].rolling(ma).min()
    df["Highest High"] = df["high"].rolling(ma).max()
    df["%K"] = (
        (df[f"{col}"] - df["Lowest Low"]) / (df["Highest High"] - df["Lowest Low"])
    ) * 100
    df["%D"] = df["%K"].rolling(threshold).mean()
    df["position"] = np.where(df["%K"] > df["%D"], 1, 0)
    return df


def minmax(df, ma, threshold, col):
    # minmax
    rolling = df[f"{col}"].rolling(ma)
    df["min"] = rolling.min()
    df["max"] = rolling.max()
    df["minmax"] = (df[f"{col}"] - df["min"]) * 2 / (df["max"] - df["min"]) - 1
    df["position"] = np.where(df["minmax"] > threshold, 1, 0)
    return df


def madiff(df, ma, threshold, col):
    """Calculate moving average difference and generate positions based on threshold."""
    rolling = df[f"{col}"].rolling(ma)
    df["ma"] = rolling.mean()
    df["madiff"] = df[f"{col}"] / df["ma"] - 1
    df["position"] = np.where(df["madiff"] > threshold, 1, 0)
    return df


def wma(df, ma, threshold, col):
    """Weighted moving average strategy."""
    weights = np.arange(1, ma + 1)
    df["wma"] = (
        df[f"{col}"]
        .rolling(ma)
        .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    )
    df["wmadiff"] = df[f"{col}"] / df["wma"] - 1
    df["position"] = np.where(df["wmadiff"] > threshold, 1, 0)
    return df


def ema(df, ma, threshold, col):
    # ema
    df["ema"] = df[f"{col}"].ewm(span=ma, adjust=False).mean()
    df["emadiff"] = df[f"{col}"] / df["ema"] - 1
    df["position"] = np.where(df["emadiff"] > threshold, 1, 0)
    return df


def percentile(df, ma, threshold, col):
    # Step 1: Calculate the percentile rank for each whales-fish
    rolling = df[f"{col}"].rolling(ma)
    df["percentile"] = rolling.rank(pct=True) * 100
    # Step 2: Scale the percentile rank to the range [-1, 1]
    df["scaled"] = 2 * (df["percentile"] / 100) - 1
    # df['position'] = np.nan
    df["position"] = np.where(df["scaled"] > threshold, 1, 0)

    return df


def vwap(df, ma, threshold, col):
    # Step 1: Calculate the percentile rank for each whales-fish
    df["hlc_vol"] = (df["high"] + df["low"] + df[f"{col}"]) / 3 * df["volume"]
    rolling = df["hlc_vol"].rolling(ma).sum()
    rolling_vol = df["volume"].rolling(ma).sum()

    df["vwap"] = rolling / rolling_vol
    # df['position'] = np.nan
    df["position"] = np.where(df["vwap"] > threshold, 1, 0)

    return df


def log(df, ma, threshold, col):
    # Ensure no negative or zero values in whales-fish (avoid log errors)
    if (df[f"{col}"] <= 0).any():
        raise ValueError("The column must contain strictly positive values.")

    # Log transformation
    df["log"] = np.log(df[f"{col}"] + 1)
    df["ma"] = df["log"].rolling(ma).mean()
    # Create position signals based on the threshold and moving average
    df["position"] = np.where(
        df[f"{col}"] > df["ma"] * (1 + threshold),
        1,
        np.where(df[f"{col}"] < df["ma"] * (1 - threshold), -1, 0),
    )

    return df


def boxcox(df, ma, threshold, col):
    # Ensure the data is positive by shifting if necessary
    if (df[f"{col}"] <= 0).any():
        shift = abs(df[f"{col}"].min()) + 1  # Shift to make all values positive
        df["long_short_ratio_shifted"] = df[f"{col}"] + shift
    else:
        df["long_short_ratio_shifted"] = df[f"{col}"]

    # Apply Box-Cox transformation
    if ma != 0:
        df["boxcox"] = (df["long_short_ratio_shifted"] ** ma - 1) / ma
    else:
        df["boxcox"] = np.log(df["long_short_ratio_shifted"])

    # Create position signals based on the threshold
    df["position"] = np.where(df["boxcox"] > threshold, 1, 0)

    return df
