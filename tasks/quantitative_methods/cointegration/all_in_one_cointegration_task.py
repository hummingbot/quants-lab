import asyncio
import logging
import os
from datetime import timedelta

from core.task_base import BaseTask

from tasks.data_collection.funding_rates_task import FundingRatesTask
from tasks.quantitative_methods.cointegration.cointegration_task import CointegrationTask
from tasks.quantitative_methods.cointegration.stat_arb_config_generator_task import StatArbConfigGeneratorTask


class AllInOneTask(BaseTask):
    def __init__(self, name: str, frequency: str, config: dict):
        super().__init__(name, frequency, config)

    async def execute(self):
        try:
            coint_task = CointegrationTask("cointegration_task",
                                           frequency=timedelta(hours=1),
                                           config=self.config["cointegration"])
            funding_task = FundingRatesTask("funding_rate_task",
                                            frequency=timedelta(hours=1),
                                            config=self.config["funding_rate"])
            stat_arb_config_gen_task = StatArbConfigGeneratorTask("stat_arb_config_generator_task",
                                                                  frequency=timedelta(hours=1),
                                                                  config=self.config["stat_arb_config_generator"])
            await coint_task.execute()
            await funding_task.execute()
            await stat_arb_config_gen_task.execute()
        except Exception as e:
            logging.error(e)


async def main():
    config = {
        "cointegration": {
            "connector_names": ["binance_perpetual"],
            "quote_asset": "USDT",
            "mongo_uri": os.getenv("MONGO_URI", ""),
            "candles_config": {
                "connector_name": "binance_perpetual",
                "interval": "15m",
                "days": 14,
                "batch_size": 20,
                "sleep_time": 5.0
            },
            "update_candles": False,
            "volume_quantile": 0.75,
            "z_score_threshold": 0.5,
            "lookback_days": 14,
            "signal_days": 3,
            "p_value_threshold": 0.05,
            "entry_threshold": 1.5,
            "stop_threshold": 1.0,
            "grid_levels": 5,
            "time_limit_hours": 24,
            "start_price_multiplier": 0.1,
            "limit_price_multiplier": 0.2,
            "end_price_multiplier": 0.0,
        },
        "funding_rate": {
            "connector_names": ["binance_perpetual"],
            "quote_asset": "USDT",
            "mongo_uri": os.getenv("MONGO_URI", ""),
        },
        "stat_arb_config_generator": {
            "mongo_uri": os.getenv("MONGO_URI", "")
        }
    }
    task = AllInOneTask(name="golden_task", config=config, frequency=timedelta(days=1))
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())