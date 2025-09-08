import asyncio
import logging
import os
import time
from datetime import timedelta
import pandas as pd
from dotenv import load_dotenv

from core.services.mongodb_client import MongoClient
from core.task_base import BaseTask


class StatArbConfigGeneratorTask(BaseTask):
    def __init__(self, name: str, frequency: str, config: dict):
        super().__init__(name, frequency, config)
        self.mongo_client = MongoClient(uri=self.config.get("mongo_uri", ""),
                                        database="quants_lab")

    async def initialize(self):
        await self.mongo_client.connect()

    @property
    def base_config(self):
        return {
            "total_amount_quote": 1000,
            "connector_name": "binance_perpetual",
            "order_frequency": 5,
            "min_order_amount_quote": 5,
            "leverage": 50,
            "min_spread_between_orders": 0.0002,
            "max_open_orders": 3,
            "max_orders_per_batch": 1,
            "activation_bounds": 0.0003,
            "open_order_type": 3,
            "stop_loss": 0.1,
            "stop_loss_order_type": 1,
            "take_profit": 0.0008,
            "take_profit_order_type": 3,
            "time_limit": 259200,
            "time_limit_order_type": 1,
            "activation_price": 0.03,
            "trailing_delta": 0.005
        }

    def get_config_dict(
            self,
            base: str,
            quote: str,
            base_start_price: float,
            base_end_price: float,
            base_limit_price: float,
            base_beta: float,
            quote_start_price: float,
            quote_end_price: float,
            quote_limit_price: float,
            quote_beta: float,
    ):
        config_dict = {
            "id": f"{base.replace('-', '')}_{quote.replace('-', '')}_coint_config",
            "controller_name": "stat_arb",
            "controller_type": "generic",
            "total_amount_quote": self.base_config.get("total_amount_quote", 1000),
            "manual_kill_switch": None,
            "connector_name": self.base_config.get("connector_name", "binance_perpetual"),
            "base_trading_pair": base,
            "quote_trading_pair": quote,
            "base_side": 1,
            "grid_config_base": {
                "end_price": base_end_price if base_end_price > base_start_price else base_start_price,
                "limit_price": base_limit_price if base_limit_price > 0 else 0.0,
                "start_price": base_start_price if base_start_price < base_end_price else base_end_price,
                # "beta": base_beta,
                "order_frequency": self.base_config.get("order_frequency", 5),
                "min_order_amount_quote": self.base_config.get("min_order_amount_quote", 25)
            },
            "grid_config_quote": {
                "end_price": quote_end_price if quote_end_price > quote_start_price else quote_start_price,
                "limit_price": quote_limit_price if quote_limit_price > 0 else 0.0,
                "start_price": quote_start_price if quote_start_price < quote_end_price else quote_end_price,
                # "beta": quote_beta,
                "order_frequency": self.base_config.get("order_frequency", 5),
                "min_order_amount_quote": self.base_config.get("min_order_amount_quote", 25)
            },
            "leverage": self.base_config.get("leverage", 50),
            "position_mode": "HEDGE",
            "min_spread_between_orders": self.base_config.get("min_spread_between_orders", 0.0002),
            "max_open_orders": self.base_config.get("max_open_orders", 3),
            "max_orders_per_batch": self.base_config.get("max_orders_per_batch", 1),
            "activation_bounds": self.base_config.get("activation_bounds", 0.0003),
            "safe_extra_spread": 0.0002,
            "deduct_base_fees": False,
            "triple_barrier_config": {
                "open_order_type": self.base_config.get("open_order_type", 3),
                "stop_loss": self.base_config.get("stop_loss", 0.1),
                "stop_loss_order_type": self.base_config.get("stop_loss_order_type", 1),
                "take_profit": self.base_config.get("take_profit", 0.0008),
                "take_profit_order_type": self.base_config.get("take_profit_order_type", 3),
                "time_limit": self.base_config.get("time_limit", 259200),
                "time_limit_order_type": self.base_config.get("time_limit_order_type", 1),
                "trailing_stop": {
                    "activation_price": self.base_config.get("activation_price", 0.03),
                    "trailing_delta": self.base_config.get("trailing_delta", 0.005),
                }
            }
        }
        return config_dict

    async def execute(self):
        """
        1) Read from mongo db funding rates
        2) Read from mongo db cointegration results
        3) Generate and store configs in MongoDB
        """
        try:
            await self.initialize()
            funding_rates = await self.mongo_client.get_documents("funding_rates_processed")
            funding_rates_df = pd.DataFrame(funding_rates)
            funding_rates_df = funding_rates_df[funding_rates_df["timestamp"] == funding_rates_df["timestamp"].max()]
            coint_results = await self.mongo_client.get_documents("cointegration_results")
            coint_results_df = pd.DataFrame(coint_results)
            results_df_1 = coint_results_df.merge(funding_rates_df, left_on=["quote", "base"], right_on=["pair1", "pair2"], how="inner")
            results_df_2 = coint_results_df.merge(funding_rates_df, left_on=["base", "quote"], right_on=["pair1", "pair2"], how="inner")
            df = pd.concat([results_df_1, results_df_2])

            # Explode the grid_base columns
            df = pd.concat([
                df.drop(['grid_base', 'grid_quote'], axis=1),
                df['grid_base'].apply(pd.Series).add_prefix('base_'),
                df['grid_quote'].apply(pd.Series).add_prefix('quote_')
            ], axis=1)

            # Generate configs
            all_configs = []
            for _, row in df.iterrows():
                record = {
                    "config": self.get_config_dict(
                        base=row["base"],
                        quote=row["quote"],
                        base_start_price=row["base_start_price"],
                        base_end_price=row["base_end_price"],
                        base_limit_price=row["base_limit_price"],
                        base_beta=row["base_beta"],
                        quote_start_price=row["quote_start_price"],
                        quote_end_price=row["quote_end_price"],
                        quote_limit_price=row["quote_limit_price"],
                        quote_beta=row["quote_beta"]
                    ),
                    "extra_info": {
                        "coint_value": row["coint_value"],
                        "rate_difference": row["rate_difference"],
                        "base_rate": row["rate1"],
                        "base_beta": row["base_beta"],
                        "base_p_value": row["base_p_value"],
                        "base_z_score": row["base_z_score"],
                        "base_side": row["base_side"],
                        "base_signal_strength": row["base_signal_strength"],
                        "base_mean_reversion_prob": row["base_mean_reversion_prob"],
                        "quote_rate": row["rate2"],
                        "quote_beta": row["quote_beta"],
                        "quote_p_value": row["quote_p_value"],
                        "quote_z_score": row["quote_z_score"],
                        "quote_side": row["quote_side"],
                        "quote_signal_strength": row["quote_signal_strength"],
                        "quote_mean_reversion_prob": row["quote_mean_reversion_prob"],
                    },
                    "timestamp": time.time()
                }
                all_configs.append(record)

            # Store configs in MongoDB
            await self.mongo_client.insert_documents(collection_name="controller_configs",
                                                     documents=all_configs,
                                                     index=[("controller_name", 1),
                                                            ("controller_type", 1),
                                                            ("connector_name", 1)])
            logging.info(f"Successfully stored {len(all_configs)} trading configs")

        except Exception as e:
            logging.error(f"Error executing golden task: {str(e)}")
            raise


async def main():
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI", "")
    config = {
        "mongo_uri": mongo_uri
    }
    task = StatArbConfigGeneratorTask(name="golden_task", frequency=timedelta(hours=12), config=config)
    await task.execute()

if __name__ == "__main__":
    asyncio.run(main())
