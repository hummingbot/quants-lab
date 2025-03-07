import time
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import patch

import pandas as pd
import numpy as np
import os
import subprocess
import plotly.graph_objects as go
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.executors.grid_executor.grid_executor import GridExecutor
from hummingbot.strategy_v2.executors.position_executor.data_types import TripleBarrierConfig, TrailingStop
from plotly.subplots import make_subplots

from core.data_sources import CLOBDataSource


def fetch_dbs(root_path: str):
    backend_host = os.getenv("BACKEND_API_SERVER")
    backend_user = os.getenv("BACKEND_API_SERVER_USER")
    backend_data_path = os.getenv("BACKEND_API_SERVER_DATA_PATH")
    local_path = os.path.join(root_path, "data/live_bot_databases")

    # Ensure local directory exists
    os.makedirs(local_path, exist_ok=True)
    # Step 1: List directories in BACKEND_API_SERVER_DATA_PATH
    list_dirs_cmd = f'ssh {backend_user}@{backend_host} "ls -d {backend_data_path}/*/"'
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
        find_files_cmd = f'ssh {backend_user}@{backend_host} "find {data_folder} -type f -name \'*.sqlite\' ! -name \'v2_with_controllers.sqlite\'"'

        try:
            file_result = subprocess.run(find_files_cmd, shell=True, capture_output=True, text=True, check=True)
            sqlite_files = file_result.stdout.strip().split("\n")
        except subprocess.CalledProcessError as e:
            print(f"Error fetching SQLite files in {data_folder}: {e.stderr}")
            continue  # Skip this folder if there's an error

        # Step 3: Transfer SQLite files using SCP
        for remote_file in sqlite_files:
            if remote_file:  # Ignore empty results
                scp_cmd = f"scp {backend_user}@{backend_host}:{remote_file} {local_path}/"
                try:
                    subprocess.run(scp_cmd, shell=True, check=True)
                    print(f"Downloaded: {remote_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to fetch {remote_file}: {e.stderr}")


def calculate_performance_metrics(selected_controller: str, all_trades_df: pd.DataFrame, side: int = 1):
    performance_df = pd.DataFrame(all_trades_df[all_trades_df["controller_id"] == selected_controller])
    performance_df.sort_values("timestamp", inplace=True)
    performance_df = performance_df[performance_df["side"] == side]
    performance_df["datetime"] = pd.to_datetime(performance_df["timestamp"], unit="s")

    # Initialize columns
    performance_df["base_amount_open"] = np.where(performance_df["position_action"] == "OPEN", performance_df["base_amount"], 0)
    performance_df["base_amount_close"] = np.where(performance_df["position_action"] == "CLOSE", performance_df["base_amount"], 0)
    performance_df["cum_base_open"] = performance_df["base_amount_open"].cumsum()
    performance_df["cum_base_close"] = performance_df["base_amount_close"].cumsum()

    performance_df["cum_quote_open"] = (performance_df["base_amount_open"] * performance_df["price"]).cumsum()
    performance_df["cum_quote_close"] = (performance_df["base_amount_close"] * performance_df["price"]).cumsum()

    # Break-even calculation
    performance_df["break_even_open"] = performance_df["cum_quote_open"] / performance_df["cum_base_open"]
    performance_df["break_even_close"] = performance_df["cum_quote_close"] / performance_df["cum_base_close"]

    # PnL calculations
    if side == 1:  # Long
        performance_df["realized_pnl"] = (performance_df["break_even_close"] - performance_df["break_even_open"]) * performance_df["cum_base_close"]
        performance_df["unrealized_pnl"] = (performance_df["price"] - performance_df["break_even_open"]) * (performance_df["cum_base_open"] - performance_df["cum_base_close"])
    else:  # Short (side=2)
        performance_df["realized_pnl"] = (performance_df["break_even_open"] - performance_df["break_even_close"]) * performance_df["cum_base_close"]
        performance_df["unrealized_pnl"] = (performance_df["break_even_open"] - performance_df["price"]) * (performance_df["cum_base_open"] - performance_df["cum_base_close"])

    # Global PnL
    performance_df["global_pnl"] = performance_df["realized_pnl"] + performance_df["unrealized_pnl"] - performance_df["cumulative_fee_paid_quote"].cumsum()
    return performance_df


async def get_execution_candles(performance_df: pd.DataFrame):
    clob = CLOBDataSource()
    candles = await clob.get_candles(connector_name="binance_perpetual",
                                     trading_pair=performance_df["trading_pair"].iloc[0],
                                     interval="1m",
                                     start_time=performance_df["timestamp"].min() - 5 * 60,
                                     end_time=performance_df["timestamp"].max() + 5 * 60)
    return candles


async def plot_candles_with_global_pnl_chart(df: pd.DataFrame, side: int = 1):
    candles = await get_execution_candles(df)
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


async def create_coint_figure(connector_instance,
                              controller_config,
                              base_candles,
                              quote_candles,
                              extra_info,
                              plot_prices: bool = False):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[controller_config["base_trading_pair"], controller_config["quote_trading_pair"]],
        x_title="Time",
        y_title="Price"
    )

    # Add base market candlesticks
    fig.add_trace(
        go.Scatter(
            x=base_candles.index,
            y=base_candles["close"],
            mode="lines",
            name=f"{controller_config['base_trading_pair']} Close"
        ),
        row=1, col=1
    )

    # Add quote market candlesticks
    fig.add_trace(
        go.Scatter(
            x=quote_candles.index,
            y=quote_candles["close"],
            mode="lines",
            name=f"{controller_config['quote_trading_pair']} Close"
        ),
        row=2, col=1
    )

    # Add horizontal lines for the base market
    fig.add_hline(
        y=controller_config["grid_config_base"]["start_price"],
        row=1, col=1,
        line=dict(color="green", width=2)
    )
    fig.add_hline(
        y=controller_config["grid_config_base"]["end_price"],
        row=1, col=1,
        line=dict(color="green", width=2)
    )
    fig.add_hline(
        y=controller_config["grid_config_base"]["limit_price"],
        row=1, col=1,
        line=dict(color="green", dash="dash", width=2)
    )
    if plot_prices:
        # Add prices for base market
        base_prices, _ = await get_executor_prices(controller_config, connector_instance=connector_instance)
        for price in base_prices:
            fig.add_hline(
                y=price,
                row=1, col=1,
                line=dict(color="gray", dash="dash")
            )

        # Add prices for quote market
        quote_prices, _ = await get_executor_prices(controller_config, side="short", connector_instance=connector_instance)
        for price in quote_prices:
            fig.add_hline(
                y=price,
                row=2, col=1,
                line=dict(color="gray", dash="dash")
            )

    # Add horizontal lines for the quote market
    fig.add_hline(
        y=controller_config["grid_config_quote"]["start_price"],
        row=2, col=1,
        line=dict(color="red", width=2)
    )
    fig.add_hline(
        y=controller_config["grid_config_quote"]["end_price"],
        row=2, col=1,
        line=dict(color="red", width=2)
    )
    fig.add_hline(
        y=controller_config["grid_config_quote"]["limit_price"],
        row=2, col=1,
        line=dict(color="red", dash="dash", width=2)
    )
    fig.add_vline(pd.to_datetime(extra_info["timestamp"], unit="s"))

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0.1)',
        font={"color": 'white', "size": 12},
        height=400,  # Smaller height for thumbnails
        hovermode="x unified",
        showlegend=False
    )
    return fig


async def apply_filters(connector_instance,
                        config: Dict[str, Any],
                        base_entry_price: float,
                        quote_entry_price: float,
                        max_base_step: float,
                        max_quote_step: float,
                        min_grid_range_ratio: float,
                        max_grid_range_ratio: float,
                        max_entry_price_distance: float,
                        max_notional_size: float = 20.0):
    if base_entry_price is None or quote_entry_price is None:
        return False

    # Calculate base grid metrics
    base_start_price = config["grid_config_base"]["start_price"]
    base_end_price = config["grid_config_base"]["end_price"]
    base_executor_prices, base_step = await get_executor_prices(config, connector_instance=connector_instance)
    base_level_amount_quote = config["total_amount_quote"] / len(base_executor_prices)

    base_grid_range_pct = base_end_price / base_start_price - 1
    base_entry_price_distance_from_start = (base_entry_price / base_start_price - 1) / base_grid_range_pct

    # Calculate quote grid metrics
    quote_start_price = config["grid_config_quote"]["start_price"]
    quote_end_price = config["grid_config_quote"]["end_price"]
    quote_executor_prices, quote_step = await get_executor_prices(config, side="short", connector_instance=connector_instance)
    quote_level_amount_quote = config["total_amount_quote"] / len(quote_executor_prices)

    quote_grid_range_pct = quote_end_price / quote_start_price - 1
    quote_entry_price_distance_from_start = 1 - (quote_entry_price / quote_start_price - 1) / quote_grid_range_pct

    # Conditions using the input values
    base_step_condition = base_step <= max_base_step
    quote_step_condition = quote_step <= max_quote_step
    grid_range_gt_zero_condition = base_grid_range_pct > 0 and quote_grid_range_pct > 0
    grid_range_pct_condition = min_grid_range_ratio <= (
            base_grid_range_pct / quote_grid_range_pct) <= max_grid_range_ratio
    base_entry_price_condition = base_entry_price_distance_from_start < max_entry_price_distance
    quote_entry_price_condition = quote_entry_price_distance_from_start < max_entry_price_distance
    inside_grid_condition = ((base_start_price < base_entry_price < base_end_price) and
                             (quote_start_price < quote_entry_price < quote_end_price))
    price_non_zero_condition = (base_end_price > 0 and quote_end_price > 0 and base_start_price > 0
                                and quote_start_price > 0 and base_end_price > 0 and quote_end_price > 0)
    # TODO: this should be applied after adjusting config proposals
    notional_size_condition = ((base_level_amount_quote <= max_notional_size) and
                               (quote_level_amount_quote <= max_notional_size))
    return (base_step_condition and quote_step_condition and grid_range_pct_condition and
            base_entry_price_condition and inside_grid_condition and price_non_zero_condition and
            grid_range_gt_zero_condition and quote_entry_price_condition and notional_size_condition)


def get_grid_executor_config(controller_config_dict: Dict[str, Any], side: str = "long"):
    side_config = "grid_config_base" if side == "long" else "grid_config_quote"
    return GridExecutorConfig(id=controller_config_dict["id"],
                              type="generic",
                              timestamp=time.time(),
                              leverage=controller_config_dict["leverage"],
                              connector_name=controller_config_dict["connector_name"],
                              trading_pair=controller_config_dict["base_trading_pair"],
                              start_price=controller_config_dict[side_config]["start_price"],
                              end_price=controller_config_dict[side_config]["end_price"],
                              limit_price=controller_config_dict[side_config]["limit_price"],
                              side=1 if side == "long" else 2,
                              total_amount_quote=controller_config_dict["total_amount_quote"] / 2,
                              min_spread_between_orders=controller_config_dict["min_spread_between_orders"],
                              min_order_amount_quote=controller_config_dict[side_config]["min_order_amount_quote"],
                              max_open_orders=controller_config_dict["max_open_orders"],
                              max_orders_per_batch=controller_config_dict["max_orders_per_batch"],
                              order_frequency=controller_config_dict[side_config]["order_frequency"],
                              activation_bounds=controller_config_dict["activation_bounds"],
                              triple_barrier_config=TripleBarrierConfig(
                                  stop_loss=controller_config_dict["triple_barrier_config"]["stop_loss"],
                                  take_profit=controller_config_dict["triple_barrier_config"]["take_profit"],
                                  time_limit=controller_config_dict["triple_barrier_config"]["time_limit"],
                                  trailing_stop=TrailingStop(
                                      activation_price=controller_config_dict["triple_barrier_config"]["trailing_stop"]["activation_price"],
                                      trailing_delta=controller_config_dict["triple_barrier_config"]["trailing_stop"]["trailing_delta"])))


async def get_executor_prices(executor_config_dict: Dict[str, Any],
                              connector_instance,
                              side: str = "long"):
    executor_config = get_grid_executor_config(executor_config_dict, side=side)
    connector_name = executor_config.connector_name
    with patch.object(connector_instance, "get_price_by_type", return_value=Decimal(str(executor_config.start_price))):
        controller = GridExecutor(ScriptStrategyBase({connector_name: connector_instance}), config=executor_config)
    prices = [level.price for level in controller.grid_levels]
    return prices, controller.step


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
