import logging
import os
import subprocess
import warnings

from typing import Dict, Any, List

from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px

import plotly.graph_objects as go
import numpy as np
import pandas as pd

from core.data_sources import CLOBDataSource
from core.data_sources.hummingbot_database import HummingbotDatabase
from core.data_structures.candles import Candles
from core.performance.models import TradingSession
from core.services.mongodb_client import MongoClient

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")


class PerformanceReport:
    def __init__(self, mongo_uri: str, database: str, from_timestamp: float, to_timestamp: float,
                 root_path: str, backend_host: str, backend_user: str, backend_data_path: str, owner: str):
        self.mongo_client = MongoClient(uri=mongo_uri, database=database)
        self.clob = CLOBDataSource()
        self.from_timestamp = from_timestamp
        self.to_timestamp = to_timestamp
        self.root_path = root_path
        self.backend_host = backend_host
        self.backend_user = backend_user
        self.backend_data_path = backend_data_path
        self.owner = owner
        self.dbs = []
        self.all_controllers = []
        self.all_executors_df = pd.DataFrame()
        self.all_configs = {}
        self.all_trades_df = pd.DataFrame()
        self.trading_sessions = []
        self.trading_pairs = []

    async def initialize(self, fetch_dbs: bool = False):
        await self.mongo_client.connect()
        if fetch_dbs:
            self.fetch_dbs(root_path=self.root_path,
                           host=self.backend_host,
                           user=self.backend_user,
                           data_path=self.backend_data_path)

    @property
    def summary(self):
        return (f"Total instances: {len(self.dbs)}" + "\n"
                f"Total configs: {len(self.all_configs)}")

    @property
    def controller_name(self):
        raise NotImplementedError

    async def load_data(self):
        all_controllers = []
        all_executors = pd.DataFrame()
        self.dbs = self.list_dbs()
        for db_name in self.dbs:
            try:
                db = HummingbotDatabase(db_name=db_name, root_path=self.root_path)
                executors_df = db.get_executors_data()
                executors_df["db_name"] = db_name
                valid_controllers = self.get_all_controllers(db)
                valid_controllers_ids = [controller["id"] for controller in valid_controllers]
                all_executors = pd.concat(
                    [all_executors, executors_df[executors_df["controller_id"].isin(valid_controllers_ids)]])
                all_controllers.extend(valid_controllers)
            except Exception as e:
                print(e)
                continue
        all_configs = {controller["id"]: controller for controller in all_controllers}

        self.all_configs = all_configs
        self.all_controllers = all_controllers
        self.all_executors_df = all_executors
        self.all_trades_df = self.get_all_trades_df()
        self.trading_pairs = list(self.all_trades_df["trading_pair"].unique())

    def get_all_controllers(self, db: HummingbotDatabase) -> List[Dict[str, Any]]:
        valid_controllers = []
        controllers = db.get_controller_data().to_dict(orient="records")
        for controller in controllers:
            if controller["config"]["controller_name"] == self.controller_name:
                controller["db_name"] = db.db_name
                controller["session_id"] = f"{db.db_name}_{controller['id']}"
                valid_controllers.append(controller)
        return valid_controllers

    def list_dbs(self) -> List[str]:
        return [db_path for db_path in os.listdir(os.path.join(self.root_path, "data", "live_bot_databases"))
                if db_path != ".gitignore"]

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
        return all_trades_df

    async def build_trading_sessions(self) -> List[TradingSession]:
        raise NotImplementedError

    @staticmethod
    def calculate_performance_fields(trades_df: pd.DataFrame, controller_id: str = None, side: int = 1):
        performance_df = trades_df.copy()
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

    @staticmethod
    def summarize_performance_metrics(df: pd.DataFrame, controller_config: Dict[str, Any], side: int = 1):
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
            "total_duration_minutes": total_duration_minutes,
            "side": "long" if side == 1 else "short",
        }
        return metrics

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

    def create_instances_gantt_chart(self):
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
            texttemplate="<b>%{label}</b><br>Vol: %{value:,}"
        )

        # Optimize layout to reduce empty space
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),  # Remove extra margins
            autosize=True,
            height=600,  # Adjust height
            width=800  # Adjust width
        )
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
