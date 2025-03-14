from decimal import Decimal
from typing import Dict, List, Optional, Set

from pydantic import Field
from pydantic.main import BaseModel

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import OrderType, PositionMode, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import TrailingStop, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


class GridLimitsConfig(BaseModel):
    start_price: Decimal
    end_price: Decimal
    limit_price: Decimal
    min_order_amount_quote: Decimal = Decimal("10")
    order_frequency: int = 5



class StatArbConfig(ControllerConfigBase):
    """
    Configuration for the Statistical Arbitrage strategy controller.
    Creates paired grid positions - long on one market and short on another.
    """
    controller_type = "generic"
    controller_name: str = "stat_arb"
    coerce_tp_to_step: bool = True
    candles_config: List[CandlesConfig] = []

    # Market Configuration
    connector_name: str = Field(default="binance_perpetual", client_data=ClientFieldData(is_updatable=True))
    base_trading_pair: str = Field("BTC-USDT", client_data=ClientFieldData(is_updatable=True))
    quote_trading_pair: str = Field("LTC-USDT", client_data=ClientFieldData(is_updatable=True))
    base_side: TradeType = Field(TradeType.BUY, client_data=ClientFieldData(is_updatable=True))
    grid_config_base: GridLimitsConfig = Field(GridLimitsConfig(
        start_price=Decimal("92000"),
        end_price=Decimal("100000"),
        limit_price=Decimal("90000")
    ), client_data=ClientFieldData(is_updatable=True))
    grid_config_quote: GridLimitsConfig = Field(GridLimitsConfig(
        start_price=Decimal("86"),
        end_price=Decimal("115"),
        limit_price=Decimal("120")
    ), client_data=ClientFieldData(is_updatable=True))

    # Account Configuration
    leverage: int = Field(default=20, client_data=ClientFieldData(is_updatable=True))
    position_mode: PositionMode = PositionMode.HEDGE

    # Grid Parameters for both markets
    total_amount_quote: Decimal = Field(default=Decimal("1000"), client_data=ClientFieldData(is_updatable=True))
    min_spread_between_orders: Decimal = Field(default=Decimal("0.0005"),
                                               client_data=ClientFieldData(is_updatable=True))
    max_open_orders: int = Field(default=5, client_data=ClientFieldData(is_updatable=True))
    max_orders_per_batch: Optional[int] = Field(default=None, client_data=ClientFieldData(is_updatable=True))
    activation_bounds: Optional[Decimal] = Field(default=None, client_data=ClientFieldData(is_updatable=True))
    safe_extra_spread: Decimal = Field(default=Decimal("0.0002"), client_data=ClientFieldData(is_updatable=True))
    deduct_base_fees: bool = Field(default=False, client_data=ClientFieldData(is_updatable=True))

    # Risk Management
    triple_barrier_config: TripleBarrierConfig = TripleBarrierConfig(
        take_profit=Decimal("0.0008"),
        stop_loss=Decimal("0.1"),
        time_limit=60 * 60 * 72,  # 3 days
        open_order_type=OrderType.LIMIT_MAKER,
        take_profit_order_type=OrderType.LIMIT_MAKER,
        stop_loss_order_type=OrderType.MARKET,
        trailing_stop=TrailingStop(activation_price=Decimal("0.03"), trailing_delta=Decimal("0.005"))
    )

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.base_trading_pair)
        markets[self.connector_name].add(self.quote_trading_pair)
        return markets


class StatArb(ControllerBase):
    def __init__(self, config: StatArbConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self._executors_created = False
        rates_required = [ConnectorPair(connector_name=config.connector_name, trading_pair=config.base_trading_pair),
                          ConnectorPair(connector_name=config.connector_name, trading_pair=config.quote_trading_pair)]
        self.market_data_provider.initialize_rate_sources(rates_required)

    def active_executors(self) -> List[ExecutorInfo]:
        return [
            executor for executor in self.executors_info
            if executor.is_active
        ]

    def determine_executor_actions(self) -> List[ExecutorAction]:
        if self._executors_created:
            return []

        self._executors_created = True
        current_time = self.market_data_provider.time()

        # Create grid executors for both pairs
        return [
            # Base market executor
            CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=GridExecutorConfig(
                    timestamp=current_time,
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.base_trading_pair,
                    side=self.config.base_side,
                    start_price=self.config.grid_config_base.start_price,
                    end_price=self.config.grid_config_base.end_price,
                    limit_price=self.config.grid_config_base.limit_price,
                    leverage=self.config.leverage,
                    total_amount_quote=self.config.total_amount_quote / 2,
                    min_spread_between_orders=self.config.min_spread_between_orders,
                    min_order_amount_quote=self.config.grid_config_base.min_order_amount_quote,
                    max_open_orders=self.config.max_open_orders,
                    max_orders_per_batch=self.config.max_orders_per_batch,
                    order_frequency=self.config.grid_config_base.order_frequency,
                    activation_bounds=self.config.activation_bounds,
                    safe_extra_spread=self.config.safe_extra_spread,
                    triple_barrier_config=self.config.triple_barrier_config,
                    deduct_base_fees=self.config.deduct_base_fees,
                    coerce_tp_to_step=self.config.coerce_tp_to_step,
                )
            ),
            # Quote market executor (opposite side)
            CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=GridExecutorConfig(
                    timestamp=current_time,
                    connector_name=self.config.connector_name,
                    trading_pair=self.config.quote_trading_pair,
                    # Opposite side of the base market
                    side=TradeType.SELL if self.config.base_side == TradeType.BUY else TradeType.BUY,
                    start_price=self.config.grid_config_quote.start_price,
                    end_price=self.config.grid_config_quote.end_price,
                    limit_price=self.config.grid_config_quote.limit_price,
                    leverage=self.config.leverage,
                    total_amount_quote=self.config.total_amount_quote / 2,
                    min_spread_between_orders=self.config.min_spread_between_orders,
                    min_order_amount_quote=self.config.grid_config_quote.min_order_amount_quote,
                    max_open_orders=self.config.max_open_orders,
                    max_orders_per_batch=self.config.max_orders_per_batch,
                    order_frequency=self.config.grid_config_quote.order_frequency,
                    activation_bounds=self.config.activation_bounds,
                    safe_extra_spread=self.config.safe_extra_spread,
                    triple_barrier_config=self.config.triple_barrier_config,
                    deduct_base_fees=self.config.deduct_base_fees,
                    coerce_tp_to_step=self.config.coerce_tp_to_step
                )
            )
        ]

    async def update_processed_data(self):
        """Update any processed data required by the controller."""
        pass

    def to_format_status(self) -> List[str]:
        lines = []
        lines.append("\n")
        lines.append("Statistical Arbitrage Grid Trading")
        lines.append("═" * 60)

        # Configuration
        lines.append("Pair Configuration:")
        lines.append(f"  Base Market: {self.config.base_trading_pair} ({self.config.base_side})")
        lines.append(f"  Quote Market: {self.config.quote_trading_pair} "
                     f"({'SELL' if self.config.base_side == TradeType.BUY else 'BUY'})")
        lines.append(f"Grid Size: {self.config.total_amount_quote}")
        lines.append("─" * 60)

        # Active Positions
        lines.append("Position Status:")
        for executor in self.active_executors():
            lines.append(f"  {executor.trading_pair}:")
            lines.append(f"    Status: {executor.status}")
            if hasattr(executor, "custom_info"):
                pnl = executor.custom_info.get("realized_pnl_quote", Decimal("0"))
                position = executor.custom_info.get("position_size_quote", Decimal("0"))
                lines.append(f"    Position Size: {position:.4f}")
                lines.append(f"    Realized PnL: {pnl:.4f}")

        lines.append("═" * 60)
        return lines
