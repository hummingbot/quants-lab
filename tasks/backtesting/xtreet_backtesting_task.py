import asyncio
import datetime
import logging
import os
from datetime import timedelta
from dotenv import load_dotenv

from core.backtesting.optimizer import StrategyOptimizer
from core.task_base import BaseTask
from research_notebooks.xtreet_bb.xtreet_bt import XtreetBacktesting
from research_notebooks.xtreet_bb.xtreet_config_gen_simple import XtreetConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class XtreetBacktestingTask(BaseTask):
    async def execute(self):
        resolution = self.config["resolution"]
        kwargs = {
            "root_path": self.config["root_path"],
            "db_host": self.config["optuna_config"]["host"],
            "db_port": self.config["optuna_config"]["port"],
            "db_user": self.config["optuna_config"]["user"],
            "db_pass": self.config["optuna_config"]["password"],
            "database_name": self.config["optuna_config"]["database"],
        }
        storage_name = StrategyOptimizer.get_storage_name(
            engine=self.config.get("engine", "sqlite"),
            **kwargs)
        for trading_pair in self.config.get("selected_pairs"):
            connector_name = self.config.get("connector_name")
            custom_backtester = XtreetBacktesting()
            optimizer = StrategyOptimizer(
                storage_name=storage_name,
                resolution=resolution,
                root_path=self.config["root_path"],
                custom_backtester=custom_backtester,
            )

            optimizer.load_candles_cache_by_connector_pair(connector_name, trading_pair)
            candles_1s = optimizer._backtesting_engine._bt_engine.backtesting_data_provider.candles_feeds[
                (f"{connector_name}_{trading_pair}_{resolution}")]
            start_date = candles_1s.index.min()
            end_date = candles_1s.index.max()
            logger.info(f"Optimizing strategy for {connector_name} {trading_pair} {start_date} {end_date}")
            config_generator = XtreetConfigGenerator(start_date=start_date, end_date=end_date)
            config_generator.trading_pair = trading_pair
            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
            await optimizer.optimize(study_name=f"{self.config['study_name']}_{today_str}",
                                     config_generator=config_generator, n_trials=50)


async def main():
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "localhost"),
        "port": os.getenv("OPTUNA_PORT", 5433),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "resolution": "1s",
        "optuna_config": optuna_config,
        "connector_name": "binance_perpetual",
        "selected_pairs": ["1000BONK-USDT"],
        "engine": "sqlite",
        "study_name": "xtreet_test_4",
        # "selected_pairs": ['1000SHIB-USDT', 'WLD-USDT', 'ACT-USDT', '1000BONK-USDT', 'DOGE-USDT', 'AGLD-USDT',
        #                    'SUI-USDT', '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT',
        #                    'HMSTR-USDT', 'TROY-USDT', '1000PEPE-USDT', '1000X-USDT', 'PNUT-USDT', 'SOL-USDT',
        #                    'XRP-USDT', 'SWELL-USDT']
    }

    task = XtreetBacktestingTask("Backtesting", timedelta(hours=12), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
