import asyncio
import datetime
import logging
import os
import sys
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
sys.path.append(root_path)  # noqa: E402

import pandas as pd
from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.backtesting.executor_simulator_base import ExecutorSimulation
from hummingbot.strategy_v2.backtesting.executors_simulator.position_executor_simulator import (
    PositionExecutorConfig,
    PositionExecutorSimulator,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop  # noqa: E402
from hummingbot.strategy_v2.models.executors import CloseType
from hummingbot.strategy_v2.utils.distributions import Distributions  # noqa: E402

from controllers.market_making.pmm_simple import PMMSimpleConfig  # noqa: E402
from core.backtesting import BacktestingEngine  # noqa: E402
from core.data_sources.clob import CLOBDataSource  # noqa: E402

# Load local dataset
# The historical data lives under the repository's `history` folder.
# The original path pointed to `nexus/history`, which does not contain the
# data file shipped with the repository, so we point directly to the
# repository-level `history` folder instead.
local_data_path = os.path.join(root_path, "history", "binance-futures")
print(f"Using local data path: {local_data_path}")
backtesting = BacktestingEngine(load_cached_data=False)

# The default backtesting engine attempts to retrieve trading rules from
# the exchange via network calls. The environment used for the example
# execution has no network access, so we monkey patch the provider
# methods that would otherwise try to reach the exchange.
provider = backtesting._bt_engine.backtesting_data_provider

async def _noop_initialize_trading_rules(connector_name: str):
    return None

# Patch PositionExecutorSimulator to use label-based slicing to avoid
# pandas FutureWarning when indexing with timestamp values.
def _patched_simulate(self: PositionExecutorSimulator, df: pd.DataFrame, config: PositionExecutorConfig, trade_cost: float):
    if config.triple_barrier_config.open_order_type.is_limit_type():
        entry_condition = (df['close'] <= config.entry_price) if config.side == TradeType.BUY else (df['close'] >= config.entry_price)
        start_timestamp = df[entry_condition]['timestamp'].min()
    else:
        start_timestamp = df['timestamp'].min()
    last_timestamp = df['timestamp'].max()

    tp = float(config.triple_barrier_config.take_profit) if config.triple_barrier_config.take_profit else None
    trailing_sl_trigger_pct = None
    trailing_sl_delta_pct = None
    if config.triple_barrier_config.trailing_stop:
        trailing_sl_trigger_pct = float(config.triple_barrier_config.trailing_stop.activation_price)
        trailing_sl_delta_pct = float(config.triple_barrier_config.trailing_stop.trailing_delta)
    tl = config.triple_barrier_config.time_limit if config.triple_barrier_config.time_limit else None
    tl_timestamp = config.timestamp + tl if tl else last_timestamp

    df_filtered = df.loc[:tl_timestamp].copy()

    df_filtered['net_pnl_pct'] = 0.0
    df_filtered['net_pnl_quote'] = 0.0
    df_filtered['cum_fees_quote'] = 0.0
    df_filtered['filled_amount_quote'] = 0.0
    df_filtered['current_position_average_price'] = float(config.entry_price)

    if pd.isna(start_timestamp):
        return ExecutorSimulation(config=config, executor_simulation=df_filtered, close_type=CloseType.TIME_LIMIT)

    entry_price = df.loc[start_timestamp, 'close']
    side_multiplier = 1 if config.side == TradeType.BUY else -1

    returns_df = df_filtered.loc[start_timestamp:]
    returns = returns_df['close'].pct_change().fillna(0)
    cumulative_returns = (((1 + returns).cumprod() - 1) * side_multiplier) - trade_cost
    df_filtered.loc[start_timestamp:, 'net_pnl_pct'] = cumulative_returns
    df_filtered.loc[start_timestamp:, 'filled_amount_quote'] = float(config.amount) * entry_price
    df_filtered['net_pnl_quote'] = df_filtered['net_pnl_pct'] * df_filtered['filled_amount_quote']
    df_filtered['cum_fees_quote'] = trade_cost * df_filtered['filled_amount_quote']

    if trailing_sl_trigger_pct is not None and trailing_sl_delta_pct is not None:
        df_filtered.loc[(df_filtered['net_pnl_pct'] > trailing_sl_trigger_pct).cummax(), 'ts'] = (
            df_filtered['net_pnl_pct'] - float(trailing_sl_delta_pct)
        ).cummax()

    first_tp_timestamp = df_filtered[df_filtered['net_pnl_pct'] > tp]['timestamp'].min() if tp else None
    first_sl_timestamp = None
    if config.triple_barrier_config.stop_loss:
        sl = float(config.triple_barrier_config.stop_loss)
        sl_price = entry_price * (1 - sl * side_multiplier)
        sl_condition = df_filtered['low'] <= sl_price if config.side == TradeType.BUY else df_filtered['high'] >= sl_price
        first_sl_timestamp = df_filtered[sl_condition]['timestamp'].min()
    first_trailing_sl_timestamp = (
        df_filtered[(~df_filtered['ts'].isna()) & (df_filtered['net_pnl_pct'] < df_filtered['ts'])]['timestamp'].min()
        if trailing_sl_delta_pct and trailing_sl_trigger_pct else None
    )
    close_timestamp = min([
        timestamp
        for timestamp in [first_tp_timestamp, first_sl_timestamp, tl_timestamp, first_trailing_sl_timestamp]
        if not pd.isna(timestamp)
    ])

    if close_timestamp == first_tp_timestamp:
        close_type = CloseType.TAKE_PROFIT
    elif close_timestamp == first_sl_timestamp:
        close_type = CloseType.STOP_LOSS
    elif close_timestamp == first_trailing_sl_timestamp:
        close_type = CloseType.TRAILING_STOP
    else:
        close_type = CloseType.TIME_LIMIT

    df_filtered = df_filtered.loc[:close_timestamp]
    df_filtered.loc[df_filtered.index[-1], 'filled_amount_quote'] = df_filtered['filled_amount_quote'].iloc[-1] * 2

    simulation = ExecutorSimulation(
        config=config,
        executor_simulation=df_filtered,
        close_type=close_type,
    )
    return simulation

PositionExecutorSimulator.simulate = _patched_simulate

provider.initialize_trading_rules = _noop_initialize_trading_rules
provider.quantize_order_amount = lambda connector_name, trading_pair, amount: amount
provider.quantize_order_price = lambda connector_name, trading_pair, price: price

config = PMMSimpleConfig(
    connector_name="binance_perpetual",
    trading_pair="BTCUSDT",
    sell_spreads=Distributions.arithmetic(3, 0.002, 0.001),
    buy_spreads=Distributions.arithmetic(3, 0.002, 0.001),
    buy_amounts_pct=[1, 1, 1],
    sell_amounts_pct=[1, 1, 1],
    total_amount_quote=Decimal("1000"),
    take_profit=Decimal("0.003"),
    stop_loss=Decimal("0.003"),
    trailing_stop=TrailingStop(
        activation_price=Decimal("0.001"), trailing_delta=Decimal("0.0005")
    ),
    time_limit=60 * 60,
    cooldown_time=60 * 60,
    executor_refresh_time=60,
)

start = int(datetime.datetime(2024, 1, 1).timestamp())
end = int(datetime.datetime(2024, 1, 2).timestamp())

async def main():
    clob = CLOBDataSource(local_data_path=local_data_path)
    try:
        # Preload candles from local file
        candles_df = clob._load_local_dataset("binance_perpetual", "BTCUSDT", "1m")
        if candles_df is None:
            raise FileNotFoundError(
                f"Candles file BTCUSDT_1m.parquet not found in {local_data_path}."
            )
        backtesting._bt_engine.backtesting_data_provider.candles_feeds["binance_perpetual_BTCUSDT_1m"] = candles_df
        #backtesting._bt_engine.backtesting_data_provider.start_time = candles_df["timestamp"].min()
        #backtesting._bt_engine.backtesting_data_provider.end_time = candles_df["timestamp"].max()

        backtesting._bt_engine.backtesting_data_provider.start_time = start
        backtesting._bt_engine.backtesting_data_provider.end_time = end

        result = await backtesting.run_backtesting(config, start, end, "1m")
        print(result.get_results_summary())
    finally:
        await clob.trades_feeds["binance_perpetual"]._session.close()


if __name__ == "__main__":
    asyncio.run(main())
