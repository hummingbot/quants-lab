import asyncio
import os
import time
from datetime import timedelta
from typing import List, Dict, Any
from research_notebooks.statarb_v2.stat_arb_performance_utils import get_executor_prices
from tasks.deployment.deployment_base_task import DeploymentBaseTask, ConfigCandidate


class StatArbDeploymentTask(DeploymentBaseTask):

    async def _fetch_controller_configs(self) -> List[ConfigCandidate]:
        min_timestamp = time.time() - self.config.get("min_config_timestamp", 24 * 60 * 60)
        controller_configs_query = {"timestamp": {"$gt": min_timestamp}}
        controller_configs_data = await self.mongo_client.get_documents(collection_name="controller_configs",
                                                                        query=controller_configs_query)
        return [ConfigCandidate.from_mongo(config_data) for config_data in controller_configs_data]

    def _extract_trading_pairs(self, config_candidates: List[ConfigCandidate]):
        return {
            candidate.config["base_trading_pair"] for candidate in config_candidates} | {
            candidate.config["quote_trading_pair"] for candidate in config_candidates
        }

    def _filter_configs_by_trading_pair(self, all_config_candidates: List[ConfigCandidate], trading_pairs: List[str]):
        return [
            candidate for candidate in all_config_candidates
            if candidate.config["base_trading_pair"] in trading_pairs
            or candidate.config["quote_trading_pair"] in trading_pairs
        ]

    async def _is_candidate_valid(self, candidate: ConfigCandidate, filter_params: Dict[str, Any]):
        # Step 1: extract config params from task config
        base_entry_price = self.last_prices.get(candidate.config["base_trading_pair"])
        quote_entry_price = self.last_prices.get(candidate.config["quote_trading_pair"])
        max_base_step = self.config["filter_candidate_params"].get("max_base_step", 0.001)
        max_quote_step = self.config["filter_candidate_params"].get("max_quote_step", 0.001)
        min_grid_range_ratio = self.config["filter_candidate_params"].get("min_grid_range_ratio", 0.5)
        max_grid_range_ratio = self.config["filter_candidate_params"].get("max_grid_range_ratio", 2.0)
        max_entry_price_distance = self.config["filter_candidate_params"].get("max_entry_price_distance", 0.4)
        max_notional_size = self.config["filter_candidate_params"].get("max_notional_size", 20.0)

        # [Optional] Step 2: add high level stop conditions
        if base_entry_price is None or quote_entry_price is None:
            return False

        # Step 3: Calculate filter conditions
        config = candidate.config.copy()

        base_start_price = config["grid_config_base"]["start_price"]
        base_end_price = config["grid_config_base"]["end_price"]
        base_executor_prices, base_step = await get_executor_prices(config, connector_instance=self.connector_instance)
        base_level_amount_quote = config["total_amount_quote"] / len(base_executor_prices)

        base_grid_range_pct = base_end_price / base_start_price - 1
        base_entry_price_distance_from_start = (base_entry_price / base_start_price - 1) / base_grid_range_pct

        quote_start_price = config["grid_config_quote"]["start_price"]
        quote_end_price = config["grid_config_quote"]["end_price"]
        quote_executor_prices, quote_step = await get_executor_prices(config, side="short",
                                                                      connector_instance=self.connector_instance)
        quote_level_amount_quote = config["total_amount_quote"] / len(quote_executor_prices)

        quote_grid_range_pct = quote_end_price / quote_start_price - 1
        quote_entry_price_distance_from_start = 1 - (
                    quote_entry_price / quote_start_price - 1) / quote_grid_range_pct

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
        # Step 4: Return boolean value
        return (base_step_condition and quote_step_condition and grid_range_pct_condition and
                base_entry_price_condition and inside_grid_condition and price_non_zero_condition and
                grid_range_gt_zero_condition and quote_entry_price_condition and notional_size_condition)

    def _adjust_config_candidates(self, config_candidates: List[ConfigCandidate]):
        params = self.config["config_adjustment_params"]
        for candidate in config_candidates:
            year, iso_week = self.get_year_and_isoweek()
            time_formatted = f"{year}||isoweek{iso_week}"
            connector_name = candidate.config["connector_name"]
            base_trading_pair = candidate.config["base_trading_pair"]
            quote_trading_pair = candidate.config["quote_trading_pair"]
            controller_id = f"{connector_name.replace('_', '-')}||{base_trading_pair}||{quote_trading_pair}||{time_formatted}"
            tag = self.get_rounded_time()
            candidate.config["id"] = f"{controller_id}_{tag}"

            candidate.config["total_amount_quote"] = params["total_amount_quote"]
            candidate.config["coerce_tp_to_step"] = params["coerce_tp_to_step"]

            base_min_order_amount_quote = float(self.min_notionals_dict[base_trading_pair]) * 1.5
            quote_min_order_amount_quote = float(self.min_notionals_dict[quote_trading_pair]) * 1.5
            min_order_amount_quote = max(base_min_order_amount_quote, quote_min_order_amount_quote)
            candidate.config["grid_config_base"]["min_order_amount_quote"] = min_order_amount_quote
            candidate.config["grid_config_quote"]["min_order_amount_quote"] = min_order_amount_quote

            candidate.config["leverage"] = params["leverage"]
            candidate.config["connector_name"] = self.config["connector_name"]
            candidate.config["min_spread_between_orders"] = params["min_spread_between_orders"]
            candidate.config["triple_barrier_config"] = {
                'stop_loss': params["stop_loss"],
                'take_profit': params["take_profit"],
                'time_limit': params["time_limit"],
                'trailing_stop': {
                    'activation_price': params["activation_price"],
                    'trailing_delta': params["trailing_delta"]
                }
            }
        return config_candidates


async def main():
    connector_name = "binance_perpetual"
    mongo_uri = (
        f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME', 'admin')}:"
        f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'admin')}@"
        f"{os.getenv('MONGO_HOST', 'localhost')}:"
        f"{os.getenv('MONGO_PORT', '27017')}/"
    )
    task_config = {
        "connector_name": connector_name,
        "mongo_uri": mongo_uri,
        "backend_api_server": os.getenv("BACKEND_API_SERVER", "localhost"),
        "min_config_timestamp": 1.5 * 24 * 60 * 60,
        "filter_candidate_params": {
            "max_base_step": 0.001,
            "max_quote_step": 0.001,
            "min_grid_range_ratio": 0.5,
            "max_grid_range_ratio": 2.0,
            "max_entry_price_distance": 0.5,
            "max_notional_size": 100.0
        },
        "config_adjustment_params": {
            "total_amount_quote": 1000.0,
            "min_spread_between_orders": 0.0004,
            "leverage": 50,
            "time_limit": 259200,
            "stop_loss": 0.1,
            "trailing_delta": 0.005,
            "take_profit": 0.0008,
            "activation_price": 0.03,
            "coerce_tp_to_step": True,
        },
        "deploy_params": {
            "max_bots": 1,
            "max_controller_configs": 2,
            "script_name": "v2_with_controllers.py",
            "image_name": "hummingbot/hummingbot:latest",
            "credentials": "master_account",
            "time_to_cash_out": 2 * 24 * 60 * 60,
        },
        "control_params": {
            "controller_max_drawdown": 0.005,
            "controller_max_pnl": 0.005,
            "global_time_limit": 24 * 60 * 60,
            "partial_drawdown": 0.1,
            "partial_profit": 0.1,
            "min_early_stop_time": 6 * 60 * 60,
            "max_early_stop_time": 24 * 60 * 60
        }
    }
    task = StatArbDeploymentTask(name="deployment_task",
                                 frequency=timedelta(minutes=20),
                                 config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
