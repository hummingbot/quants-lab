import asyncio
import logging
import os
import sys
import warnings

from datetime import datetime
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from core.data_sources import CLOBDataSource
from core.data_structures.candles import Candles
from core.performance.models import TradingSession
from core.performance.performance_report import PerformanceReport

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


class StatArbTradingSession(TradingSession):
    def to_serializable_dict(self) -> dict:
        """
        Convert to dict with serializable types (e.g. DataFrames to dicts).
        """
        data = self.dict()
        perf = data["performance_metrics"]

        for key in ["long_df", "short_df"]:
            if key in perf and isinstance(perf[key], pd.DataFrame):
                perf[key] = perf[key].to_dict(orient="list")

        return data

    @classmethod
    def from_serializable_dict(cls, data: dict) -> "TradingSession":
        """
        Create TradingSession from serialized dict (e.g. dicts back to DataFrames).
        """
        perf = data.get("performance_metrics", {})
        for key in ["long_df", "short_df"]:
            if key in perf and isinstance(perf[key], dict):
                perf[key] = pd.DataFrame(perf[key])
        return cls(**data)


class StatArbPerformanceReport(PerformanceReport):
    def __init__(self, mongo_uri: str, database: str, from_timestamp: float, to_timestamp: float,
                 root_path: str, backend_host: str, backend_user: str, backend_data_path: str, owner: str):
        super().__init__(mongo_uri, database, from_timestamp, to_timestamp, root_path, backend_host,
                         backend_user, backend_data_path, owner)
        self.cointegration_df = pd.DataFrame()
        self.cointegration_df_filtered = pd.DataFrame()

    @property
    def controller_name(self):
        return "stat_arb"

    async def initialize(self, fetch_dbs: bool = False):
        await super().initialize(fetch_dbs=fetch_dbs)
        self.cointegration_df = await self.get_cointegration_df()

    async def get_cointegration_df(self):
        query = {
            "timestamp": {
                "$gt": self.from_timestamp,
                "$lt": self.to_timestamp
            }
        }
        cointegration_analysis = await self.mongo_client.get_documents(collection_name="cointegration_results",
                                                                       query=query)
        cointegration_values = []
        for document in cointegration_analysis:
            timestamp = document["timestamp"]
            base_asset = document["base"]
            quote_asset = document["quote"]
            coint_value = document["coint_value"]
            base_beta = document["grid_base"]["beta"]
            base_p_value = document["grid_base"]["p_value"]
            base_z_score = document["grid_base"]["z_score"]
            base_side = document["grid_base"]["side"]
            base_start_price = document["grid_base"]["start_price"]
            base_end_price = document["grid_base"]["end_price"]
            quote_beta = document["grid_quote"]["beta"]
            quote_p_value = document["grid_quote"]["p_value"]
            quote_z_score = document["grid_quote"]["z_score"]
            quote_side = document["grid_quote"]["side"]
            quote_start_price = document["grid_quote"]["start_price"]
            quote_end_price = document["grid_quote"]["end_price"]

            cointegration_values.append(
                {
                    "timestamp": timestamp,
                    "base_asset": base_asset,
                    "quote_asset": quote_asset,
                    "coint_value": coint_value,
                    "base_beta": base_beta,
                    "base_p_value": base_p_value,
                    "base_z_score": base_z_score,
                    "base_side": base_side,
                    "base_start_price": base_start_price,
                    "base_end_price": base_end_price,
                    "quote_beta": quote_beta,
                    "quote_p_value": quote_p_value,
                    "quote_z_score": quote_z_score,
                    "quote_side": quote_side,
                    "quote_start_price": quote_start_price,
                    "quote_end_price": quote_end_price,
                }
            )
        cointegration_df = pd.DataFrame(cointegration_values)
        self.cointegration_df = cointegration_df
        return cointegration_df

    async def build_trading_sessions(self) -> List[StatArbTradingSession]:
        trading_sessions = []
        for controller in self.all_controllers:
            controller_id = controller["id"]
            session_id = controller["session_id"]
            long_metrics = short_metrics = {}
            try:
                controller_config = controller["config"]
                controller_config["id"] = controller_id
                long_df = self.calculate_performance_fields(trades_df=self.all_trades_df,
                                                            controller_id=controller_id,
                                                            side=1)
                short_df = self.calculate_performance_fields(trades_df=self.all_trades_df,
                                                             controller_id=controller_id,
                                                             side=2)

                if len(long_df) > 0:
                    long_metrics = self.summarize_performance_metrics(long_df, controller_config, side=1)
                if len(short_df) > 0:
                    short_metrics = self.summarize_performance_metrics(short_df, controller_config, side=2)
                start_timestamp = self.get_valid_timestamp(long_df, short_df, how="min")
                end_timestamp = self.get_valid_timestamp(long_df, short_df, how="max")
                coint_info = self.get_cointegration_info(controller_config, start_timestamp)
                performance_metrics = {
                    "long_df": long_df,
                    "long_metrics": long_metrics,
                    "short_df": short_df,
                    "short_metrics": short_metrics,
                    "coint_info": coint_info
                }
                trading_sessions.append(StatArbTradingSession(session_id=session_id,
                                                              db_name=controller["db_name"],
                                                              start_timestamp=start_timestamp,
                                                              end_timestamp=end_timestamp,
                                                              controller_config=controller_config,
                                                              performance_metrics=performance_metrics))
            except Exception as e:
                print(f"Error generating trading session for {controller_id}: {e}")
                continue
        self.trading_sessions = trading_sessions
        return trading_sessions

    @staticmethod
    def get_valid_timestamp(long_df: pd.DataFrame, short_df: pd.DataFrame, how="min") -> Optional[pd.Timestamp]:
        if how == "min":
            long_timestamp = long_df.timestamp.min()
            short_timestamp = short_df.timestamp.min()
        elif how == "max":
            long_timestamp = long_df.timestamp.max()
            short_timestamp = short_df.timestamp.max()
        else:
            raise ValueError("Only 'min' or 'max' values are allowed.")

        if pd.notna(long_timestamp) and pd.notna(short_timestamp):
            return min(long_timestamp, short_timestamp)
        if pd.notna(long_timestamp):
            return long_timestamp
        if pd.notna(short_timestamp):
            return short_timestamp
        return None

    async def upload_trading_sessions(self):
        trading_sessions = []
        for session in self.trading_sessions:
            session_id = session.session_id
            start_timestamp = session.start_timestamp
            end_timestamp = session.end_timestamp
            status = self._determine_status(session)
            session_dict = {
                "id": session_id,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "owner": self.owner,
                "controller_name": self.controller_name,
                "trading_session": session.to_serializable_dict(),
                "status": status
            }
            trading_sessions.append(session_dict)
        try:
            await self.mongo_client.insert_documents(collection_name="trading_sessions",
                                                     documents=trading_sessions,
                                                     db_name="quants_lab")
        except Exception as e:
            logging.error(f"Couldn't upload documents: {e}")

    @staticmethod
    def _determine_status(session: StatArbTradingSession):
        long_df_cond = len(session.performance_metrics["long_df"]) > 0
        short_df_cond = len(session.performance_metrics["short_df"]) > 0
        if long_df_cond and short_df_cond:
            return {"success": True, "msg": ""}
        if not long_df_cond and not short_df_cond:
            return {"success": False, "msg": "Missing long or short data"}
        if not long_df_cond:
            return {"success": False, "msg": "Missing long data"}
        if not short_df_cond:
            return {"success": False, "msg": "Missing short data"}
        return {"success": True, "msg": "Success"}

    async def get_execution_candles(self, performance_df: pd.DataFrame) -> Candles:
        self.clob = CLOBDataSource()
        candles = await self.clob.get_candles(connector_name="binance_perpetual",
                                              trading_pair=performance_df["trading_pair"].iloc[0],
                                              interval="1m",
                                              start_time=performance_df["timestamp"].min() - 5 * 60,
                                              end_time=performance_df["timestamp"].max() + 5 * 60)
        return candles

    def get_cointegration_info(self, controller_config: Dict[str, Any], to_timestamp: float):
        df = self.cointegration_df.copy()
        if len(df) > 0:
            long_trading_pair = controller_config["base_trading_pair"]
            short_trading_pair = controller_config["quote_trading_pair"]
            coint_values = df[(df["base_asset"] == long_trading_pair) & (df["quote_asset"] == short_trading_pair) &
                              (df["timestamp"] <= to_timestamp)].sort_values("timestamp")
            if len(coint_values) > 0:
                coint_info = coint_values.iloc[-1].to_dict()
                coint_info["signal_delay_hours"] = (to_timestamp - coint_info["timestamp"]) / 3600
                return coint_info
        return {}

    def create_cointegration_heatmap(self, export: bool = False):
        df = pd.DataFrame(
            [session.performance_metrics["coint_info"] for session in self.trading_sessions])

        df.dropna(subset=["base_asset", "quote_asset"], inplace=True)

        df["timestamp"] = pd.to_numeric(df["timestamp"])

        # Keep only the last timestamp per (base_asset, quote_asset) pair
        df_filtered = df.loc[df.groupby(["base_asset", "quote_asset"])["timestamp"].idxmax()]

        # Reset index (optional)
        df_filtered = df_filtered[(df_filtered["base_asset"].isin(self.trading_pairs)) &
                                  (df_filtered["quote_asset"].isin(self.trading_pairs))]

        # Create a symmetric matrix
        cointegration_matrix = df_filtered.pivot(index="base_asset", columns="quote_asset", values="coint_value")
        cointegration_matrix = cointegration_matrix.combine_first(cointegration_matrix.T)

        # Fill diagonal with 1.0 (or NaN if preferred)
        for asset in set(df_filtered["base_asset"]).union(df_filtered["quote_asset"]):
            cointegration_matrix.loc[asset, asset] = 1.0

        # Get median value for white midpoint
        median_value = np.nanmedian(cointegration_matrix.values)

        np.fill_diagonal(cointegration_matrix.values, np.nan)

        # Define custom colorscale
        custom_colorscale = [
            [0.0, "green"],  # Low values (close to 0) → Green
            [0.5, "white"],  # Median value → White
            [1.0, "red"]  # High values → Red
        ]

        # Replace NaNs in annotations with an empty string
        annotations = np.where(pd.isna(cointegration_matrix.values), "", np.round(cointegration_matrix.values, 2))

        # Create Heatmap with Annotations
        fig = go.Figure(data=go.Heatmap(
            z=cointegration_matrix.values,
            x=cointegration_matrix.columns,
            y=cointegration_matrix.index,
            colorscale=custom_colorscale,
            text=annotations,  # Annotate with cointegration values
            hoverinfo="text",
            texttemplate="%{text}",  # Display text in boxes
            zmid=median_value,  # Set the white midpoint for balance
        ))

        # Layout Adjustments
        fig.update_layout(
            title=f"Cointegration Analysis of Traded Markets ({len(self.trading_pairs)} Selected Pairs)",
            xaxis_title="Long Asset",
            yaxis_title="Short Asset",
            autosize=False,
            width=1200,
            height=1000,
            font=dict(size=12)
        )
        if export:
            fig.write_image("correlation_heatmap.jpg", format="jpg", scale=3)
        else:
            return fig


async def main():
    load_dotenv()
    _root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    sys.path.append(_root_path)
    params = {
        "mongo_uri": os.getenv("MONGO_URI", ""),
        "database": "quants_lab",
        "from_timestamp": datetime(2025, 3, 10, 0, 0, 0).timestamp(),
        "to_timestamp": datetime(2025, 12, 31, 23, 59, 59).timestamp(),
        "root_path": _root_path,
        "backend_host": os.getenv("BACKEND_API_SERVER"),
        "backend_user": os.getenv("BACKEND_API_SERVER_USER", "root"),
        "backend_data_path": os.getenv("BACKEND_API_SERVER_DATA_PATH", "deploy/bots/archived"),
        "owner": "drupman",
    }
    performance_report = StatArbPerformanceReport(**params)
    await performance_report.initialize(fetch_dbs=False)
    await performance_report.load_data()
    await performance_report.build_trading_sessions()
    await performance_report.upload_trading_sessions()
    print(performance_report.summary)


if __name__ == "__main__":
    asyncio.run(main())
