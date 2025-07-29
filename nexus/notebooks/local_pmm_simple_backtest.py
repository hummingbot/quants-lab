import asyncio
import datetime
import time
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
local_data_path = os.path.join(root_path, "nexus", "history", "binance-futures")
print(f"Using local data path: {local_data_path}")
backtesting = BacktestingEngine(load_cached_data=False)

# The default backtesting engine attempts to retrieve trading rules from
# the exchange via network calls. The environment used for the example
# execution has no network access, so we monkey patch the provider
# methods that would otherwise try to reach the exchange.
provider = backtesting._bt_engine.backtesting_data_provider


async def _noop_initialize_trading_rules(connector_name: str):
    return None


from core.backtesting.position_executor_patch import patch_position_executor_simulator

patch_position_executor_simulator()

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
    trailing_stop=TrailingStop(activation_price=Decimal("0.001"), trailing_delta=Decimal("0.0005")),
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
            raise FileNotFoundError(f"Candles file BTCUSDT_1m.parquet not found in {local_data_path}.")
        backtesting._bt_engine.backtesting_data_provider.candles_feeds["binance_perpetual_BTCUSDT_1m"] = candles_df
        backtesting._bt_engine.backtesting_data_provider.start_time = candles_df["timestamp"].min()
        backtesting._bt_engine.backtesting_data_provider.end_time = candles_df["timestamp"].max()

        result = await backtesting.run_backtesting(config, start, end, "1m")
        print(result.get_results_summary())
    finally:
        await clob.trades_feeds["binance_perpetual"]._session.close()

if __name__ == "__main__":
    timer_start = time.perf_counter()
    asyncio.run(main())
    timer_end = time.perf_counter()
    print(f"Elapsed just this part: {timer_end - timer_start:.4f} seconds")