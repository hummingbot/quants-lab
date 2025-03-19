import asyncio
import logging
import os
import subprocess
import sys
import warnings

from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
from pydantic.main import BaseModel

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from core.data_sources import CLOBDataSource
from core.data_sources.hummingbot_database import HummingbotDatabase
from core.data_structures.candles import Candles
from core.services.mongodb_client import MongoClient

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


class TradingSession(BaseModel):
    long_df: pd.DataFrame
    short_df: pd.DataFrame
    long_performance_fig: Optional[go.Figure]
    short_performance_fig: Optional[go.Figure]
    long_metrics: dict
    short_metrics: dict
    long_candles: pd.DataFrame
    short_candles: pd.DataFrame
    controller_config: dict
    coint_info: dict

    class Config:
        arbitrary_types_allowed = True


class StatArbPerformanceReport:
    def __init__(self, mongo_uri: str, database: str, from_timestamp: float, to_timestamp: float,
                 root_path: str, backend_host: str, backend_user: str, backend_data_path: str):
        self.mongo_client = MongoClient(uri=mongo_uri, database=database)
        self.clob = CLOBDataSource()
        self.from_timestamp = from_timestamp
        self.to_timestamp = to_timestamp
        self.root_path = root_path
        self.backend_host = backend_host
        self.backend_user = backend_user
        self.backend_data_path = backend_data_path
        self.all_controllers = []
        self.all_executors_df = pd.DataFrame()
        self.all_configs = {}
        self.all_trades_df = pd.DataFrame()
        self.trading_sessions = []
        self.trading_pairs = []
        self.cointegration_df = pd.DataFrame()
        self.cointegration_df_filtered = pd.DataFrame()

    async def initialize(self):
        await self.mongo_client.connect()

    @property
    def summary(self):
        return ""

    async def load_data(self):
        all_controllers = []
        all_executors = pd.DataFrame()
        dbs = self.list_dbs()
        for db_name in dbs:
            try:
                db = HummingbotDatabase(db_name=db_name, root_path=self.root_path)
                executors_df = db.get_executors_data()
                executors_df["db_name"] = db_name
                controllers = db.get_controller_data().to_dict(orient="records")
                valid_controllers = [controller for controller in controllers if
                                     controller["config"]["controller_name"] == "stat_arb"]
                valid_controllers_ids = [controller["id"] for controller in valid_controllers]
                all_controllers.extend(valid_controllers)
                all_executors = pd.concat(
                    [all_executors, executors_df[executors_df["controller_id"].isin(valid_controllers_ids)]])
            except Exception as e:
                print(e)
                continue
        all_configs = {controller["id"]: controller for controller in all_controllers}

        self.all_controllers = all_controllers
        self.all_executors_df = all_executors
        self.all_configs = all_configs
        self.all_trades_df = self.get_all_trades_df()
        self.cointegration_df = await self.get_cointegration_df()
        self.trading_pairs = list(self.all_trades_df["trading_pair"].unique())
        print(f"Deployed instances: {len(dbs)}")
        print(f"Configurations used: {len(all_configs)}")

    def get_all_trades_df(self):
        all_trades = []

        for _, executor in self.all_executors_df.iterrows():
            custom_info = executor["custom_info"]
            for order_filled in custom_info["filled_orders"]:
                for _, fill in order_filled["order_fills"].items():
                    trade_type = order_filled["trade_type"]  # BUY or SELL
                    position_action = order_filled["position"]  # OPEN or CLOSE

                    position_multiplier = 1 if (trade_type == "BUY" and position_action == "OPEN") or (
                                trade_type == "SELL" and position_action == "CLOSE") else -1
                    fill_dict = {
                        "db_name": executor["db_name"],
                        "controller_id": executor["controller_id"],
                        "side": executor["config"]["side"],
                        "trading_pair": order_filled["trading_pair"],
                        "order_type": order_filled["order_type"],
                        "trade_type": trade_type,
                        "cumulative_fee_paid_quote": sum(
                            [float(flat_fee["amount"]) for flat_fee in fill["fee"]["flat_fees"]]),
                        # this will be only valid when percent_token = USDT
                        "position_action": order_filled["position"],
                        "timestamp": self.ensure_timestamp_in_seconds(fill["fill_timestamp"]),
                        "price": float(fill["fill_price"]),
                        "base_amount": float(fill["fill_base_amount"]),
                        "quote_amount": float(fill["fill_quote_amount"]),
                        "position_multiplier": position_multiplier
                    }
                    all_trades.append(fill_dict)
        all_trades_df = pd.DataFrame(all_trades)

        self.all_trades_df = all_trades_df
        return all_trades_df

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

    async def build_trading_sessions(self):
        trading_sessions = []
        for controller in self.all_controllers:
            controller_id = controller["id"]
            long_candles_df = short_candles_df = pd.DataFrame()
            long_performance_fig = short_performance_fig = go.Figure()
            long_metrics = short_metrics = {}
            try:
                controller_config = controller["config"]
                controller_config["id"] = controller_id
                long_df = self.calculate_performance_fields(controller_id=controller_id, side=1)
                if len(long_df) > 0:
                    long_candles = await self.get_execution_candles(long_df)
                    long_candles_df = long_candles.data
                    long_performance_fig = await self.plot_candles_with_global_pnl_chart(long_candles, long_df, side=1)
                    long_metrics = self.summarize_performance_metrics(long_df, controller_config, side=1)
                short_df = self.calculate_performance_fields(controller_id=controller_id, side=2)
                if len(short_df) > 0:
                    short_candles = await self.get_execution_candles(short_df)
                    short_candles_df = short_candles.data
                    short_performance_fig = await self.plot_candles_with_global_pnl_chart(short_candles, short_df, side=2)
                    short_metrics = self.summarize_performance_metrics(short_df, controller_config, side=2)
                to_timestamp = long_df.timestamp.min() or short_df.timestamp.min()
                coint_info = self.get_cointegration_info(controller_config, to_timestamp)
                trading_sessions.append(TradingSession(controller_config=controller_config,
                                                       long_df=long_df,
                                                       long_performance_fig=long_performance_fig,
                                                       long_metrics=long_metrics,
                                                       long_candles=long_candles_df,
                                                       short_df=short_df,
                                                       short_performance_fig=short_performance_fig,
                                                       short_metrics=short_metrics,
                                                       short_candles=short_candles_df,
                                                       coint_info=coint_info))
            except Exception as e:
                print(f"Error generating trading session for {controller_id}: {e}")
                continue
        self.trading_sessions = trading_sessions
        self.cointegration_df_filtered = pd.DataFrame([session.coint_info for session in self.trading_sessions])
        return trading_sessions

    def calculate_performance_fields(self, controller_id: str = None, side: int = 1):
        performance_df = self.all_trades_df.copy()
        if controller_id is not None:
            performance_df = performance_df[performance_df["controller_id"] == controller_id]
        performance_df.sort_values("timestamp", inplace=True)
        performance_df = performance_df[performance_df["side"] == side]
        performance_df["datetime"] = pd.to_datetime(performance_df["timestamp"], unit="s")

        # Initialize columns
        performance_df["base_amount_open"] = np.where(performance_df["position_action"] == "OPEN",
                                                      performance_df["base_amount"], 0)
        performance_df["base_amount_close"] = np.where(performance_df["position_action"] == "CLOSE",
                                                       performance_df["base_amount"], 0)
        performance_df["cum_base_open"] = performance_df["base_amount_open"].cumsum()
        performance_df["cum_base_close"] = performance_df["base_amount_close"].cumsum()

        performance_df["cum_quote_open"] = (performance_df["base_amount_open"] * performance_df["price"]).cumsum()
        performance_df["cum_quote_close"] = (performance_df["base_amount_close"] * performance_df["price"]).cumsum()

        # Break-even calculation
        performance_df["break_even_open"] = performance_df["cum_quote_open"] / performance_df["cum_base_open"]
        performance_df["break_even_close"] = performance_df["cum_quote_close"] / performance_df["cum_base_close"]

        # PnL calculations
        if side == 1:  # Long
            performance_df["realized_pnl"] = (performance_df["break_even_close"] - performance_df["break_even_open"]) * \
                                             performance_df["cum_base_close"]
            performance_df["unrealized_pnl"] = (performance_df["price"] - performance_df["break_even_open"]) * (
                        performance_df["cum_base_open"] - performance_df["cum_base_close"])
        else:  # Short (side=2)
            performance_df["realized_pnl"] = (performance_df["break_even_open"] - performance_df["break_even_close"]) * \
                                             performance_df["cum_base_close"]
            performance_df["unrealized_pnl"] = (performance_df["break_even_open"] - performance_df["price"]) * (
                        performance_df["cum_base_open"] - performance_df["cum_base_close"])

        # Global PnL
        performance_df["global_pnl"] = (performance_df["realized_pnl"] + performance_df["unrealized_pnl"] -
                                        performance_df["cumulative_fee_paid_quote"].cumsum())
        return performance_df

    async def get_execution_candles(self, performance_df: pd.DataFrame) -> Candles:
        self.clob = CLOBDataSource()
        candles = await self.clob.get_candles(connector_name="binance_perpetual",
                                              trading_pair=performance_df["trading_pair"].iloc[0],
                                              interval="1m",
                                              start_time=performance_df["timestamp"].min() - 5 * 60,
                                              end_time=performance_df["timestamp"].max() + 5 * 60)
        return candles

    @staticmethod
    async def plot_candles_with_global_pnl_chart(candles: Candles, df: pd.DataFrame, side: int = 1):
        # Create a subplot with 2 rows
        side = "Long" if side == 1 else "Short"
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=(f"{side} OHLC Chart with Break-Even Levels", "PnL and Fees Over Time"))

        # ---------------------- FIG 1: Candlestick Chart & Break Even ----------------------
        # OHLC Candlestick Chart
        fig.add_trace(go.Candlestick(name="OHLC",
                                     x=candles.data.index,
                                     open=candles.data["open"],
                                     high=candles.data["high"],
                                     low=candles.data["low"],
                                     close=candles.data["close"]),
                      row=1, col=1)

        # Break Even Open
        fig.add_trace(go.Scatter(name="Break Even Open",
                                 x=df["datetime"],
                                 y=df["break_even_open"],
                                 marker_color="olive",
                                 line_shape="hv"),
                      row=1, col=1)

        # Break Even Close
        fig.add_trace(go.Scatter(name="Break Even Close",
                                 x=df["datetime"],
                                 y=df["break_even_close"],
                                 marker_color="red",
                                 line_shape="hv"),
                      row=1, col=1)

        # Markers for trade positions (buy/sell signals)
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(df["timestamp"], unit="s"),
            y=df["price"],
            mode="markers",
            marker=dict(
                symbol=df["position_multiplier"].apply(lambda x: "triangle-up" if x > 0 else "triangle-down"),
                size=8,
                color="white"
            ),
            showlegend=False),
            row=1, col=1
        )

        # ---------------------- FIG 2: PnL and Fees ----------------------
        # Realized PnL
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.realized_pnl,
                                 name="Realized PnL"),
                      row=2, col=1)

        # Unrealized PnL
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.unrealized_pnl,
                                 name="Unrealized PnL"),
                      row=2, col=1)

        # Global PnL (filled area)
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.global_pnl,
                                 line_shape="hv",
                                 fill="tozeroy",
                                 name="Global PnL"),
                      row=2, col=1)

        # Cumulative Fee Paid
        fig.add_trace(go.Scatter(x=df.datetime,
                                 y=df.cumulative_fee_paid_quote.cumsum(),
                                 line_shape="hv",
                                 name="Cumulative Fee"),
                      row=2, col=1)

        # ---------------------- Layout Adjustments ----------------------
        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,  # Remove range slider from first plot
            showlegend=True,
            title_text="Trading Performance Overview",
            xaxis2=dict(title="Time")  # Label x-axis only for second row
        )

        return fig

    @staticmethod
    def summarize_performance_metrics(df: pd.DataFrame, controller_config: Dict[str, Any], side: int = 1):
        # side_key = "grid_config_base" if side == 1 else "grid_config_quote"
        total_amount_quote = controller_config["total_amount_quote"]
        global_pnl = df["global_pnl"].iloc[-1]
        max_draw_down = df["global_pnl"].min() / total_amount_quote
        max_run_up = df["global_pnl"].max() / total_amount_quote
        total_trades = len(df)
        total_quote_volume = df["quote_amount"].sum()
        total_duration_minutes = (df["timestamp"].max() - df["timestamp"].min()) / 60
        metrics = {
            "trading_pair": df["trading_pair"].iloc[0],
            "global_pnl": global_pnl,
            "max_draw_down": max_draw_down,
            "max_run_up": max_run_up,
            "total_trades": total_trades,
            "total_quote_volume": total_quote_volume,
            "total_duration_minutes": total_duration_minutes
        }
        return metrics

    def get_cointegration_info(self, controller_config: Dict[str, Any], to_timestamp: float):
        df = self.cointegration_df.copy()
        long_trading_pair = controller_config["base_trading_pair"]
        short_trading_pair = controller_config["quote_trading_pair"]
        coint_values = df[(df["base_asset"] == long_trading_pair) & (df["quote_asset"] == short_trading_pair) &
                          (df["timestamp"] <= to_timestamp)].sort_values("timestamp")
        if len(coint_values) > 0:
            coint_info = coint_values.iloc[-1].to_dict()
            coint_info["signal_delay_hours"] = (to_timestamp - coint_info["timestamp"]) / 3600
            return coint_info
        return {}

    def create_gantt_chart(self):
        agg_trades_df = (
            self.all_trades_df
            .groupby(["trading_pair", "db_name"], as_index=False)
            .agg(min_timestamp=("timestamp", "min"), max_timestamp=("timestamp", "max"))
        )
        agg_executors_df = (
            self.all_executors_df
            .groupby("db_name", as_index=False)
            .agg(net_pnl_quote=("net_pnl_quote", "sum"), total_volume=("filled_amount_quote", "sum"))
        )
        global_performance = agg_trades_df.merge(agg_executors_df, on="db_name")
        global_performance["start_datetime"] = pd.to_datetime(global_performance["min_timestamp"], unit="s")
        global_performance["end_datetime"] = pd.to_datetime(global_performance["max_timestamp"], unit="s")

        # Crear Gantt chart dataframe format
        global_performance.sort_values(by="start_datetime", inplace=True)
        gantt_data = [
            dict(Task=row["db_name"], Start=row["start_datetime"], Finish=row["end_datetime"])
            for _, row in global_performance.iterrows()
        ]

        # Extraer valores únicos para definir colores
        unique_tasks = list(set(row["db_name"] for _, row in global_performance.iterrows()))

        # Generar una lista de colores lo suficientemente grande
        colors = px.colors.qualitative.Set3  # Usa una paleta con suficiente variedad
        if len(colors) < len(unique_tasks):
            colors = px.colors.qualitative.Alphabet  # Usa más colores si es necesario

        # Crear un diccionario que asigne colores a cada tarea
        color_dict = {task: colors[i % len(colors)] for i, task in enumerate(unique_tasks)}

        # Crear Gantt chart con colores dinámicos
        gantt_fig = ff.create_gantt(
            gantt_data,
            index_col="Task",
            colors=color_dict,  # Agregar colores dinámicos
            show_colorbar=False,
            group_tasks=True,
            showgrid_x=True,
            showgrid_y=True
        )

        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],  # 50%-25%-25%
            shared_xaxes=True,
            subplot_titles=[
                "Bot History Gantt",
                "Cumulative PNL Over Time",
                "Cumulative Volume Over Time"
            ],
            vertical_spacing=0.1
        )

        # Add Gantt chart traces
        for trace in gantt_fig.data:
            fig.add_trace(trace, row=1, col=1)

        global_performance.sort_values(by="end_datetime", inplace=True)
        # Add cumulative PNL scatter plot
        fig.add_trace(
            go.Scatter(
                x=global_performance["end_datetime"],
                y=global_performance["net_pnl_quote"].cumsum(),
                mode="lines+markers",
                name="Cumulative PNL",
                line=dict(color="purple")
            ),
            row=2, col=1
        )

        # Add cumulative Volume scatter plot
        fig.add_trace(
            go.Scatter(
                x=global_performance["end_datetime"],
                y=global_performance["total_volume"].cumsum(),
                mode="lines+markers",
                name="Cumulative Volume",
                line=dict(color="orange")
            ),
            row=3, col=1
        )

        # Update layout with annotations
        fig.update_layout(
            height=900,
            title_text="Bot Performance Overview",
            xaxis3_title="End DateTime",
            showlegend=False,
        )
        return fig

    def plot_volume_treemap(self):
        # Filter data
        df = self.all_trades_df.copy()

        # Round the volume column to integers
        df["quote_amount"] = df["quote_amount"].round(0).astype(int)

        # Create the treemap
        fig = px.treemap(
            df,
            path=["trading_pair"],  # No hierarchy, just trading pairs
            values="quote_amount",
            color="quote_amount",
            color_continuous_scale="viridis"
        )

        # Customize text to show volume without decimals
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>Vol: %{value:,}"  # Adds thousand separator
        )

        # Optimize layout to reduce empty space
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
            autosize=True,
            height=600,  # Adjust height
            width=800  # Adjust width
        )
        return fig

    def create_cointegration_heatmap(self, export: bool = False):
        df = self.cointegration_df_filtered.copy()
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

    @staticmethod
    def fetch_dbs(root_path: str, host: str, user: str, data_path: str):
        local_path = os.path.join(root_path, "data/live_bot_databases")

        # Ensure local directory exists
        os.makedirs(local_path, exist_ok=True)
        # Step 1: List directories in BACKEND_API_SERVER_DATA_PATH
        list_dirs_cmd = f'ssh {user}@{host} "ls -d {data_path}/*/"'
        try:
            result = subprocess.run(list_dirs_cmd, shell=True, capture_output=True, text=True, check=True)
            directories = result.stdout.strip().split("\n")
        except subprocess.CalledProcessError as e:
            print(f"Error fetching directories: {e.stderr}")
            exit(1)

        # Step 2: Iterate through directories and find SQLite files
        for directory in directories:
            sequence_dir = directory.strip()
            data_folder = f"{sequence_dir}/data"

            # Find SQLite files, excluding "v2_with_controllers.sqlite"
            find_files_cmd = f'ssh {user}@{host} "find {data_folder} -type f -name \'*.sqlite\' ! -name \'v2_with_controllers.sqlite\'"'

            try:
                file_result = subprocess.run(find_files_cmd, shell=True, capture_output=True, text=True, check=True)
                sqlite_files = file_result.stdout.strip().split("\n")
            except subprocess.CalledProcessError as e:
                print(f"Error fetching SQLite files in {data_folder}: {e.stderr}")
                continue  # Skip this folder if there's an error

            # Step 3: Transfer SQLite files using SCP
            for remote_file in sqlite_files:
                if remote_file:  # Ignore empty results
                    scp_cmd = f"scp {user}@{host}:{remote_file} {local_path}/"
                    try:
                        subprocess.run(scp_cmd, shell=True, check=True)
                        print(f"Downloaded: {remote_file}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to fetch {remote_file}: {e.stderr}")

    def list_dbs(self):
        return [db_path for db_path in os.listdir(os.path.join(self.root_path, "data", "live_bot_databases"))
                if db_path != ".gitignore"]

    @staticmethod
    def ensure_timestamp_in_seconds(timestamp: float) -> float:
        timestamp_int = int(float(timestamp))
        if timestamp_int >= 1e18:  # Nanoseconds
            return timestamp_int / 1e9
        elif timestamp_int >= 1e15:  # Microseconds
            return timestamp_int / 1e6
        elif timestamp_int >= 1e12:  # Milliseconds
            return timestamp_int / 1e3
        elif timestamp_int >= 1e9:  # Seconds
            return timestamp_int
        else:
            raise ValueError(
                "Timestamp is not in a recognized format. Must be in seconds, milliseconds, microseconds or nanoseconds.")


async def main():
    load_dotenv()
    _root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    sys.path.append(_root_path)
    fetch_dbs = False
    params = {
        "mongo_uri": os.getenv("MONGO_URI", ""),
        "database": "quants_lab",
        "from_timestamp": datetime(2025, 3, 10, 0, 0, 0).timestamp(),
        "to_timestamp": datetime(2025, 3, 16, 23, 59, 59).timestamp(),
        "root_path": _root_path,
        "backend_host": os.getenv("BACKEND_API_SERVER"),
        "backend_user": os.getenv("BACKEND_API_SERVER_USER", "root"),
        "backend_data_path": os.getenv("BACKEND_API_SERVER_DATA_PATH", "deploy/bots/archived"),
    }
    performance_report = StatArbPerformanceReport(**params)
    await performance_report.initialize()
    if fetch_dbs:
        performance_report.fetch_dbs(params["root_path"],
                                     params["backend_host"],
                                     params["backend_user"],
                                     params["backend_data_path"])
    await performance_report.load_data()
    trading_sessions = await performance_report.build_trading_sessions()
    print(performance_report.summary)


if __name__ == "__main__":
    asyncio.run(main())
