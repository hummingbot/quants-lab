from itertools import chain
from math import pi

import numpy as np
import pandas as pd
import pandas_ta as ta


class Features:
    def __init__(self, external_features = {}, df = pd.DataFrame()):
        self.external_features = external_features
        self.df = df

    # TODO: migrate this format to current core.features.candles location with feature_base approach
    def add_features(self, dropna=True):
        for key, value in self.external_features.items():
            drop = False
            for k, v in value.items():
                if k == "macd":
                    for l in v:
                        columns = self.df.columns
                        if key != "close":
                            self.df.rename(columns={key: "close"}, inplace=True)
                            self.df.ta.macd(fast=l[0], slow=l[1], signal=l[2], append=True)
                            self.df.rename(columns={"close": key}, inplace=True)
                        else:
                            self.df.ta.macd(fast=l[0], slow=l[1], signal=l[2], append=True)
                        new_cols = [c for c in self.df.columns if c not in columns]
                        for col in new_cols:
                            self.df.rename(columns={col: col + "_" + key}, inplace=True)
                elif k == "rsi":
                    for l in v:
                        columns = self.df.columns
                        if key != "close":
                            self.df.rename(columns={key: "close"}, inplace=True)
                            self.df.ta.rsi(length=l, append=True)
                            self.df.rename(columns={"close": key}, inplace=True)
                        else:
                            self.df.ta.rsi(length=l, append=True)
                        new_cols = [c for c in self.df.columns if c not in columns]
                        for col in new_cols:
                            self.df.rename(columns={col: col + "_" + key}, inplace=True)
                elif k == "bbands":
                    for l in v:
                        columns = self.df.columns
                        if key != "close":
                            self.df.rename(columns={key: "close"}, inplace=True)
                            self.df.ta.bbands(length=l[0], std=l[1], mamode=l[2], append=True)
                            self.df.rename(columns={"close": key}, inplace=True)
                        else:
                            self.df.ta.bbands(length=l[0], std=l[1], mamode=l[2], append=True)
                        new_cols = [c for c in self.df.columns if c not in columns]
                        for col in new_cols:
                            self.df.rename(columns={col: col + "_" + key}, inplace=True)
                elif k == "relative_changes":
                    for l in v:
                        self.df[key + '_relative_changes_' + str(l)] = self.df[key].pct_change(periods = l)
                elif k == "mov_avg":
                    self.add_moving_averages(column_name=key, windows=v)
                elif k == "lag":
                    self.add_lagged_values(column_name=key, lags=v)
                elif k == "volatility":
                    self.add_volatility_measures(column_name=key, windows=v)
                elif k == "drop":
                    drop = v
                elif k == "drawdowns_and_runups":
                    print(v)
                    for l in v:
                        print(l)
                        self.calculate_drawdowns_and_runups(gamma=l)

            if drop:
                self.df.drop(columns=key, inplace=True)

        string_list = ['MAC', 'BB', 'MA','volume']# List comprehension to get the columns
        columns_with_strings = [col for col in self.df.columns if any(substring in col for substring in string_list)]

        print(columns_with_strings)
        for column_name in columns_with_strings:
            self.df[column_name] = self.df[column_name].pct_change(periods = 1)
        
        if dropna:
            self.df.dropna()
        
        def replace_large_values(x):
            if isinstance(x, (int, float)):
                if np.isinf(x) or x > 9999999:
                    return 9999999
                elif x < -9999999:
                    return -9999999
            return x
        timestamp = self.df['timestamp']
        # Apply the function to the DataFrame
        self.df = self.df.applymap(replace_large_values)
        self.df['timestamp'] = timestamp
        print(self.df.describe())
        
        return self.df

    def add_moving_averages(self, column_name, windows=[7, 14, 30]):
        for window in windows:
            self.df[f'MA_{column_name}_{window}'] = self.df[column_name].rolling(window=window).mean()

    def add_lagged_values(self, column_name, lags=[1, 2, 3]):
        for lag in lags:
            self.df[f'{column_name}_lag_{lag}'] = self.df[column_name].shift(lag)

    def add_volatility_measures(self, column_name, windows=[7, 14, 30]):
        for window in windows:
            self.df[f'{column_name}_volatility_{window}'] = self.df[column_name].pct_change().rolling(window=window).std() * np.sqrt(window)

    def calculate_drawdowns_and_runups(self, gamma):
        print(self.df.index)
        self.df["peak_" + str(gamma)] = False
        self.df["peak_ascending_" + str(gamma)] = False
        ascending = True if self.df["close"].iloc[0] < self.df["open"].iloc[0] else False
        threshold = self.df["open"].iloc[0] * (1 - gamma) if ascending else self.df["open"].iloc[0] * (1 + gamma)

        current_peak = self.df["low"].iloc[0] if ascending else self.df["high"].iloc[0]

        for i in range(1, len(self.df)):
            openp = self.df["open"].iloc[i]
            high = self.df["high"].iloc[i]
            low = self.df["low"].iloc[i]
            close = self.df["close"].iloc[i]
            if ascending:
                candle_breaks_run_up_by_itself = (low - openp) / openp <= -gamma
                if candle_breaks_run_up_by_itself:
                    self.df[ "peak_" + str(gamma)].iloc[i] = True
                    current_peak = high
                    ascending = True if close > openp else False
                else:
                    if high > current_peak:
                        current_peak = high
                        threshold = current_peak * (1 - gamma)
                    if low < threshold:
                        self.df[ "peak_" + str(gamma)].iloc[i] = True
                        current_peak = high
                        ascending = True if close > openp else False
            else:
                candle_breaks_draw_down_by_itself = (high - openp) / openp >= gamma
                if candle_breaks_draw_down_by_itself:
                    self.df[ "peak_" + str(gamma)].iloc[i] = True
                    current_peak = low
                    ascending = True if close > openp else False
                else:
                    if low < current_peak:
                        current_peak = min(current_peak, low)
                        threshold = current_peak * (1 + gamma)
                    if high > threshold:
                        self.df[ "peak_" + str(gamma)].iloc[i] = True
                        current_peak = low
                        ascending = True if close > openp else False
            self.df["peak_ascending_" + str(gamma)].iloc[i] = ascending
        # self.df["peak_" + str(gamma)].replace(np.nan, False)
        # self.df["peak_" + str(gamma)] = self.df["peak_" + str(gamma)].astype(int)
        print("peak_ascending_nulls",self.df["peak_ascending_" + str(gamma)].isnull().sum())
        print("peak_nulls",self.df["peak_" + str(gamma)].isnull().sum())
        print(self.df["peak_ascending_" + str(gamma)].describe())
        print(self.df["peak_" + str(gamma)].describe())