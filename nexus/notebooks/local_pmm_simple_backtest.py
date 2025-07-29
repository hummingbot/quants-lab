import asyncio
import datetime
import os
import sys
from decimal import Decimal

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))  # noqa: E402
sys.path.append(root_path)  # noqa: E402

from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop  # noqa: E402
from hummingbot.strategy_v2.utils.distributions import Distributions  # noqa: E402

from controllers.market_making.pmm_simple import PMMSimpleConfig  # noqa: E402
from core.backtesting import BacktestingEngine  # noqa: E402
from core.data_sources.clob import CLOBDataSource  # noqa: E402

# Load local dataset
local_data_path = os.path.join(root_path, "nexus", "history", "binance-futures")

backtesting = BacktestingEngine(load_cached_data=False)

config = PMMSimpleConfig(
    connector_name="binance_perpetual",
    trading_pair="BTCUSDT",
    sell_spreads=Distributions.arithmetic(3, 0.002, 0.001),
    buy_spreads=Distributions.arithmetic(3, 0.002, 0.001),
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



if __name__ == "__main__":
    asyncio.run(main())
