from decimal import Decimal
from typing import Dict, List, Optional, Set

import pandas_ta as ta  # noqa: F401
from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo
from hummingbot.strategy_v2.utils.distributions import Distributions


class GridStrikeConfig(ControllerConfigBase):
    """
    Configuration required to run the D-Man Maker V2 strategy.
    """
    controller_name: str = "grid_strike"
    candles_config: List[CandlesConfig] = []
    connector_name: str = Field(
        default="xrpl",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the connector name: "
        ))
    trading_pair: str = Field(
        default="SOLO-XRP",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the trading pair: "
        ))
    total_amount_quote: Decimal = Field(
        default=Decimal("300"),
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the total amount in quote asset: "
        ))
    leverage: int = 1
    position_mode: PositionMode = PositionMode.HEDGE
    grid_mid_price: Decimal = Field(default=Decimal("0.16"))
    grid_upper_price: float = Field(default=0)
    grid_lower_price: float = Field(default=0.138)
    sniper_upper_price: Decimal = Field(default=Decimal("0.2"))
    sniper_lower_price: Decimal = Field(default=Decimal("0.11"))
    grid_allocation: Decimal = Field(default=Decimal(0.85))
    inventory_buffer: Decimal = Field(default=Decimal("0.05"))
    n_levels: int = Field(default=20)
    grid_step: Optional[float] = None
    take_profit: Optional[Decimal] = None
    activation_bounds: Optional[Decimal] = None
    rebalanced: bool = False

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.connector_name not in markets:
            markets[self.connector_name] = set()
        markets[self.connector_name].add(self.trading_pair)
        return markets


class GridStrike(ControllerBase):
    def __init__(self, config: GridStrikeConfig, *args, **kwargs):
        self.config = config
        self.connector_name = config.connector_name
        self.trading_pair = config.trading_pair
        if config.grid_step:
            pct_grid = ((config.grid_upper_price - config.grid_lower_price) / config.grid_lower_price)
            config.n_levels = int(pct_grid / config.grid_step) + 1
        self.grid_step = (config.grid_upper_price - config.grid_lower_price) / (config.n_levels - 1) / config.grid_lower_price
        self.take_profit = config.take_profit if config.take_profit else Decimal(self.grid_step)
        self.grid_prices = Distributions.linear(config.n_levels, config.grid_lower_price, config.grid_upper_price)
        buy_prices = [price for price in self.grid_prices if price <= config.grid_mid_price]
        sell_prices = [price for price in self.grid_prices if price > config.grid_mid_price]
        self.buy_levels = {i: price for i, price in enumerate(buy_prices)}
        self.sell_levels = {i: price for i, price in enumerate(sell_prices)}
        self.grid_allocation_quote = config.total_amount_quote * config.grid_allocation
        self.sniper_allocation_quote = config.total_amount_quote * (
                1 - config.grid_allocation - config.inventory_buffer)
        self.executor_amount_quote = self.grid_allocation_quote / config.n_levels
        self.quote_asset_allocation = (self.grid_allocation_quote * len(
            self.buy_levels) / config.n_levels) + self.sniper_allocation_quote / 2
        self.base_asset_allocation = (self.grid_allocation_quote * len(
            self.sell_levels) / config.n_levels) + self.sniper_allocation_quote / 2
        self.sniper_upper = {config.sniper_upper_price: None}
        self.sniper_lower = {config.sniper_lower_price: None}
        self.rebalanced = config.rebalanced
        super().__init__(config, *args, **kwargs)

    def active_buy_executors(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.BUY)

    def active_sell_executors(self) -> List[ExecutorInfo]:
        return self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and x.side == TradeType.SELL)

    def get_mid_price(self) -> Decimal:
        return self.market_data_provider.get_price_by_type(connector_name=self.config.connector_name,
                                                           trading_pair=self.config.trading_pair,
                                                           price_type=PriceType.MidPrice)

    def create_actions_proposal(self) -> List[ExecutorAction]:
        actions = []
        if self.rebalanced:
            actions.extend(self.create_sniper_actions())
            actions.extend(self.create_grid_actions())
        return actions

    def create_sniper_actions(self) -> List[ExecutorAction]:
        actions = []
        mid_price = self.processed_data["mid_price"]
        if len(self.processed_data["sniper_upper"]) == 0:
            take_profit = abs(self.config.sniper_upper_price - self.config.grid_mid_price) / self.config.grid_mid_price
            amount = (self.sniper_allocation_quote / Decimal("2")) / self.config.sniper_upper_price
            entry_price = self.config.sniper_upper_price if self.config.sniper_upper_price >= mid_price else mid_price
            actions.append(CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=PositionExecutorConfig(
                    timestamp=self.market_data_provider.time(),
                    trading_pair=self.trading_pair,
                    connector_name=self.connector_name,
                    side=TradeType.SELL,
                    entry_price=entry_price,
                    amount=amount,
                    leverage=self.config.leverage,
                    level_id="sniper_upper",
                    triple_barrier_config=TripleBarrierConfig(
                        take_profit=take_profit,
                    ),
                )))
        if len(self.processed_data["sniper_lower"]) == 0:
            take_profit = abs(self.config.grid_mid_price - self.config.sniper_lower_price) / self.config.sniper_lower_price
            amount = (self.sniper_allocation_quote / Decimal("2")) / self.config.sniper_lower_price
            entry_price = self.config.sniper_lower_price if self.config.sniper_lower_price <= mid_price else mid_price
            actions.append(CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=PositionExecutorConfig(
                    timestamp=self.market_data_provider.time(),
                    trading_pair=self.trading_pair,
                    connector_name=self.connector_name,
                    side=TradeType.BUY,
                    entry_price=entry_price,
                    amount=amount,
                    leverage=self.config.leverage,
                    level_id="sniper_lower",
                    triple_barrier_config=TripleBarrierConfig(
                        take_profit=take_profit,
                    ),
                )))
        return actions

    def create_grid_actions(self) -> List[ExecutorAction]:
        actions = []
        non_active_buy_grid_executors = self.processed_data["non_active_buy_grid_executors"]
        non_active_sell_grid_executors = self.processed_data["non_active_sell_grid_executors"]
        mid_price = self.processed_data["mid_price"]
        for index in non_active_buy_grid_executors:
            executor_price = self.buy_levels[index]
            if executor_price <= mid_price:
                entry_price = Decimal(executor_price)
                take_profit = self.take_profit
            else:
                entry_price = Decimal(mid_price)
                distance_pct = abs((executor_price - entry_price) / entry_price)
                take_profit = distance_pct + self.take_profit
            amount = self.executor_amount_quote / entry_price
            actions.append(CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=PositionExecutorConfig(
                    timestamp=self.market_data_provider.time(),
                    trading_pair=self.trading_pair,
                    connector_name=self.connector_name,
                    side=TradeType.BUY,
                    entry_price=entry_price,
                    amount=amount,
                    leverage=self.config.leverage,
                    level_id=index,
                    activation_bounds=[self.config.activation_bounds, Decimal("0.01")] if self.config.activation_bounds else None,
                    triple_barrier_config=TripleBarrierConfig(
                        take_profit=Decimal(take_profit),
                        open_order_type=OrderType.LIMIT_MAKER,
                        take_profit_order_type=OrderType.LIMIT_MAKER,
                    ))))
        for index in non_active_sell_grid_executors:
            executor_price = self.sell_levels[index]
            if executor_price >= mid_price:
                entry_price = Decimal(executor_price)
                take_profit = self.take_profit
            else:
                entry_price = Decimal(mid_price)
                distance_pct = abs((executor_price - entry_price) / entry_price)
                take_profit = distance_pct + self.take_profit
            amount = self.executor_amount_quote / entry_price
            actions.append(CreateExecutorAction(
                controller_id=self.config.id,
                executor_config=PositionExecutorConfig(
                    timestamp=self.market_data_provider.time(),
                    trading_pair=self.trading_pair,
                    connector_name=self.connector_name,
                    side=TradeType.SELL,
                    entry_price=entry_price,
                    amount=amount,
                    leverage=self.config.leverage,
                    level_id=index,
                    activation_bounds=[self.config.activation_bounds, Decimal("0.01")] if self.config.activation_bounds else None,
                    triple_barrier_config=TripleBarrierConfig(
                        take_profit=Decimal(take_profit),
                        open_order_type=OrderType.LIMIT_MAKER,
                        take_profit_order_type=OrderType.LIMIT_MAKER,
                    ),
                )))
        return actions

    async def update_processed_data(self):
        active_buy_executors = self.active_buy_executors()
        active_sell_executors = self.active_sell_executors()
        sniper_lower = [e for e in active_buy_executors if e.custom_info["level_id"] == "sniper_lower"]
        sniper_upper = [e for e in active_sell_executors if e.custom_info["level_id"] == "sniper_upper"]
        active_grid_buy_executors_level_ids = [int(e.custom_info["level_id"]) for e in active_buy_executors if
                                               "sniper" not in e.custom_info["level_id"]]
        active_grid_sell_executors_level_ids = [int(e.custom_info["level_id"]) for e in active_sell_executors if
                                                "sniper" not in e.custom_info["level_id"]]
        non_active_buy_grid_executors = [e for e in self.buy_levels if e not in active_grid_buy_executors_level_ids]
        non_active_sell_grid_executors = [e for e in self.sell_levels if e not in active_grid_sell_executors_level_ids]
        self.processed_data.update({
            "active_buy_executors": active_buy_executors,
            "active_sell_executors": active_sell_executors,
            "sniper_upper": sniper_upper,
            "sniper_lower": sniper_lower,
            "non_active_buy_grid_executors": non_active_buy_grid_executors,
            "non_active_sell_grid_executors": non_active_sell_grid_executors,
            "mid_price": self.get_mid_price()
        })

    def determine_executor_actions(self) -> List[ExecutorAction]:
        actions = []
        actions.extend(self.create_actions_proposal())
        return actions

    def to_format_status(self) -> List[str]:
        return []
