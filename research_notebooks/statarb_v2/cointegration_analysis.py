import asyncio
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pydantic import BaseModel
from statsmodels.tsa.stattools import coint
from typing import Tuple, List, Dict, Any
import time
import math
from dotenv import load_dotenv
import logging
import os
import pandas as pd
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from core.data_sources import CLOBDataSource
from core.data_structures.candles import Candles
from core.services.mongodb_client import MongoClient


class CointegrationV2Study(BaseModel):
    dominant: str
    hedge: str
    coint_value: float
    lookback_days_timestamp: float
    timestamp: float
    connector_name: str
    interval: str
    lookback_days: int
    current_coint_value: float
    alpha: float
    beta: float
    z_t: pd.Series
    z_mean: float
    z_std: float
    signal_strength: float
    mean_reversion_prob: float
    median_error: float
    y_pred: pd.Series
    dominant_cum_returns: pd.Series
    hedge_cum_returns: pd.Series

    class Config:
        arbitrary_types_allowed = True

    def plot_returns_with_z_score(self, candles_dict: Dict[str, Candles]):
        candles_dom = candles_dict[self.dominant].data
        candles_hed = candles_dict[self.hedge].data
        lookback_datetime = pd.to_datetime(self.lookback_days_timestamp, unit="s")
        dominant_cum_returns = candles_dom[candles_dom.index >= lookback_datetime]["close"].pct_change().add(1).cumprod().dropna()
        hedge_cum_returns = candles_hed[candles_hed.index >= lookback_datetime]["close"].pct_change().add(1).cumprod().dropna()

        row_1_title = (f"Initial Cointegration: {self.coint_value:.6f} - "
                       f"Actual Cointegration: {self.current_coint_value:.6f} - ")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[row_1_title, "Z-score"], row_heights=[2, 1], x_title="Datetime")
        fig.add_trace(go.Scatter(
            name=f"{self.dominant}",
            x=dominant_cum_returns.index,
            y=dominant_cum_returns
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            name=f"{self.hedge}",
            x=hedge_cum_returns.index,
            y=hedge_cum_returns
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            name="y_pred",
            x=self.y_pred.index,
            y=self.y_pred,
            mode='lines',
            line=dict(
                color='lightgreen',
                dash='dash'
            ),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            name="Z-score",
            x=self.z_t.index,
            y=self.z_t
        ), row=2, col=1)

        for i in [1.0, 1.5, 2.0]:
            fig.add_hline(y=i * self.z_std, row=2)
            fig.add_hline(y=-i * self.z_std, row=2)

        fig.add_vline(x=lookback_datetime)
        fig.add_vline(x=pd.to_datetime(self.timestamp, unit="s"))

        fig.update_layout(height=800, width=1400)
        return fig

    def count_crosses(self, seconds_ago: int = None):
        s = (self.dominant_cum_returns - self.hedge_cum_returns)
        if seconds_ago is not None:
            s = s[s.index > pd.to_datetime(time.time() - seconds_ago, unit="s")]
        signs = np.sign(s)
        sign_changes = (signs.shift(1) * signs) < 0
        count = sign_changes.sum()
        return count

    def get_cointegration_validation_df(self, window: int = 100):
        series = self.z_t.dropna()
        df = pd.DataFrame(index=series.index)

        # Rolling statistics
        df[f"rolling_mean_{window}"] = series.rolling(window).mean()
        df[f"rolling_median_{window}"] = series.rolling(window).median()
        df[f"rolling_std_{window}"] = series.rolling(window).std()

        # Rolling ADF p-values (start from window-th index)
        pvals = []
        pval_index = []
        for i in range(window, len(series)):
            window_data = series[i - window:i]
            pval = adfuller(window_data)[1]
            pvals.append(pval)
            pval_index.append(series.index[i])

        df.loc[pval_index, f"rolling_adf_{window}"] = pvals

        # Calculate global half-life and broadcast it
        halflife_value = self.calculate_half_life()
        df["halflife"] = halflife_value
        return df

    def calculate_half_life(self):
        spread_lag = self.z_t.shift(1)
        delta = self.z_t - spread_lag
        model = sm.OLS(delta[1:], sm.add_constant(spread_lag[1:]))
        res = model.fit()
        halflife = -np.log(2) / res.params.iloc[1]
        return halflife

    @staticmethod
    def plot_rolling_adf(cointegration_validation_df: pd.DataFrame, window: int = 100):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cointegration_validation_df.index,
                                 y=cointegration_validation_df[f"rolling_adf_{window}"],
                                 name="Rolling ADF (P-value)"))
        fig.update_layout(title="Rolling ADF (P-value)",
                          yaxis=dict(tickformat="%0.2f", title="p-value"),
                          xaxis=dict(title="datetime"))
        return fig

    @staticmethod
    def plot_rolling_stats(cointegration_validation_df: pd.DataFrame, window: int = 100):
        fig = go.Figure()
        for col in [f"rolling_mean_{window}", f"rolling_median_{window}", f"rolling_std_{window}"]:
            name = f"Rolling {col.split('_')[1].capitalize()} Z_t ({window} periods)"
            fig.add_trace(go.Scatter(name=name,
                                     x=cointegration_validation_df.index,
                                     y=cointegration_validation_df[col]))
        fig.update_layout(title="Mean and Variance Stability (z_t)",
                          yaxis=dict(title="z_t"),
                          xaxis=dict(title="datetime"),
                          legend=dict(
                              orientation="h",  # horizontal layout
                              yanchor="bottom",  # anchor at bottom of the legend box
                              y=1.02,  # position just above the plot area
                              xanchor="left",
                              x=0
                          )
                          )
        return fig


class CointegrationAnalyzer:
    def __init__(self, mongo_uri: str, database: str):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"), database="quants_lab")
        self.dominants: List[str] = None
        self.hedges: List[str] = None
        self.trading_pairs: List[str] = None
        self.interval: str = None
        self.connector_name: str = None
        self.days: int = None
        self.candles: List[Candles] = None
        self.candles_dict: Dict[str, Candles] = None
        self.z_score_threshold = 2.0
        self.analysis_cols = ["current_coint_value", "alpha", "beta", "z_t", "z_mean", "z_std", "signal_strength",
                              "mean_reversion_prob", "percentage_error", "median_error", "y_pred", "dominant_cum_returns",
                              "hedge_cum_returns", "metadata"]

    async def initialize(self):
        await self.mongo_client.connect()

    async def get_cointegration_results_v2(self):
        cointegration_docs = await self.mongo_client.get_documents("cointegration_results_v2",
                                                                   db_name="quants_lab")
        coint_docs_df = pd.DataFrame(cointegration_docs)
        last_timestamp = coint_docs_df["timestamp"].max()
        coint_docs_df = coint_docs_df[coint_docs_df["timestamp"] == last_timestamp].head(3)
        coint_docs_df["datetime"] = pd.to_datetime(coint_docs_df["timestamp"], unit="s")
        coint_docs_df["lookback_days_datetime"] = pd.to_datetime(coint_docs_df["lookback_days_timestamp"], unit="s")
        coint_docs_df.sort_values(by="coint_value", inplace=True)

        results = []
        for _, row in coint_docs_df.iterrows():
            dominant = row["direction_a"]["dominant"]
            hedge = row["direction_a"]["hedge"]
            coint_value = row["coint_value"]
            lookback_days_timestamp = row["lookback_days_timestamp"]
            lookback_days_datetime = row["lookback_days_datetime"]
            timestamp = row["timestamp"]
            datetime = row["datetime"]
            interval = row["interval"]
            connector_name = row["connector_name"]
            lookback_days = row["lookback_days"]
            metadata = row["metadata"]
            results.append({
                "dominant": dominant,
                "metadata": metadata,
                "hedge": hedge,
                "coint_value": coint_value,
                "lookback_days_timestamp": lookback_days_timestamp,
                "lookback_days_datetime": lookback_days_datetime,
                "timestamp": timestamp,
                "datetime": datetime,
                "connector_name": connector_name,
                "interval": interval,
                "lookback_days": lookback_days,
            })
        df = pd.DataFrame(results)

        self.dominants = list(df["dominant"].unique())
        self.hedges = list(df["hedge"].unique())
        self.trading_pairs = list(set(self.dominants) | set(self.hedges))
        self.interval = df["interval"].iloc[-1]
        self.connector_name = df["connector_name"].iloc[-1]
        self.days = math.ceil((time.time() - df['lookback_days_timestamp'].min()) / (24 * 60 * 60))
        return df

    async def update_candles(self):
        clob = CLOBDataSource()
        candles = await clob.get_candles_batch_last_days(connector_name=self.connector_name,
                                                         trading_pairs=self.trading_pairs,
                                                         interval=self.interval,
                                                         days=self.days)
        candles_dict = {candle.trading_pair: candle for candle in candles}
        self.candles = candles
        self.candles_dict = candles_dict

    def update_analysis_cols(self, df: pd.DataFrame):
        """
        If z_score (hedge_cum_returns - y_pred) > 0 it means that we need to short hedge and long dominant
        """
        df[self.analysis_cols] = None

        for i, row in df.iterrows():
            candles_dominant = self.candles_dict[row["dominant"]].data
            candles_hedge = self.candles_dict[row["hedge"]].data
            dominant_cum_returns = candles_dominant.loc[
                candles_dominant["timestamp"] >= row["lookback_days_timestamp"], "close"].pct_change().add(
                1).cumprod().dropna()
            hedge_cum_returns = candles_hedge.loc[
                candles_hedge["timestamp"] >= row["lookback_days_timestamp"], "close"].pct_change().add(
                1).cumprod().dropna()
            min_len = min(len(dominant_cum_returns), len(hedge_cum_returns))
            y, x = hedge_cum_returns.values[:min_len], dominant_cum_returns.values[:min_len]

            # Run Engle-Granger test
            coint_res: Tuple = coint(y, x)
            current_coint_value = coint_res[1]

            # Perform linear regression
            x_reshaped = x.reshape(-1, 1)
            reg: LinearRegression = LinearRegression().fit(x_reshaped, y)
            alpha = reg.intercept_
            beta = reg.coef_[0]

            # Calculate spread (z_t)
            z_t = pd.Series(y - (alpha + beta * x))
            z_t.index = dominant_cum_returns.index.copy()
            z_mean = z_t.mean()
            z_std = z_t.std()

            # Calculate recent predictions and spread
            y_pred = alpha + beta * dominant_cum_returns

            # Calculate current Z-score
            current_z_score = (z_t.iloc[-1] - z_mean) / z_std
            # Calculate additional metrics
            signal_strength = abs(current_z_score) / self.z_score_threshold
            mean_reversion_prob = 1 - stats.norm.cdf(abs(current_z_score))

            # Calculate percentage error for recent period
            percentage_error = ((y_pred - hedge_cum_returns) / hedge_cum_returns.abs()) * 100
            median_error = np.median(percentage_error)
            df.at[i, "current_coint_value"] = current_coint_value
            df.at[i, "alpha"] = alpha
            df.at[i, "beta"] = beta
            df.at[i, "z_t"] = z_t
            df.at[i, "z_mean"] = z_mean
            df.at[i, "z_std"] = z_std
            df.at[i, "signal_strength"] = signal_strength
            df.at[i, "mean_reversion_prob"] = mean_reversion_prob
            df.at[i, "percentage_error"] = percentage_error
            df.at[i, "median_error"] = median_error
            df.at[i, "y_pred"] = y_pred
            df.at[i, "dominant_cum_returns"] = dominant_cum_returns
            df.at[i, "hedge_cum_returns"] = hedge_cum_returns
        return df

    @staticmethod
    def create_cointegration_v2_studies(df: pd.DataFrame):
        # class_cols = [
        #     "dominant", "hedge", "coint_value", "lookback_days_timestamp", "timestamp", "connector_name",
        #     "interval", "lookback_days", "current_coint_value", "alpha", "beta", "z_t", "z_mean", "z_std",
        #     "signal_strength", "mean_reversion_prob", "median_error", "y_pred"
        # ]
        studies = []
        for _, row in df.iterrows():
            studies.append(CointegrationV2Study(**row))
        return studies


async def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    load_dotenv()
    coint_analyzer = CointegrationAnalyzer(mongo_uri=os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/"),
                                           database="cointegration_results_v2")
    await coint_analyzer.initialize()
    df = await coint_analyzer.get_cointegration_results_v2()
    await coint_analyzer.update_candles()
    df = coint_analyzer.update_analysis_cols(df)
    studies = coint_analyzer.create_cointegration_v2_studies(df)
    print(f"Successfully built {len(studies)} cointegration studies.")


if __name__ == "__main__":
    asyncio.run(main())
