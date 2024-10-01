import asyncio
import datetime
import logging
import os
from datetime import timedelta
from typing import Any, Dict

from core.backtesting.optimizer import StrategyOptimizer
from core.task_base import BaseTask
from core.utils import load_dict_from_yaml
from research_notebooks.xtreet_bb.xtreet_config_generator import XtreetConfigGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestingTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.root_path = config.get('root_path', "")

    async def execute(self):
        start_date = datetime.datetime.now() - timedelta(days=self.config.get('lookback_days', 1))
        end_date = datetime.datetime.now()
        logger.info("Generating config generator")
        config_generator = XtreetConfigGenerator(start_date=start_date, end_date=end_date)
        logger.info("Generating top markets report")
        await config_generator.generate_top_markets_report(self.config["config"])
        logger.info("Optimizing strategy")
        optimizer = StrategyOptimizer(root_path=self.root_path, resolution=self.config.get('resolution', "1s"))
        await optimizer.optimize_custom_configs(study_name=self.config.get('study_name', "xtreet_1"),
                                                config_generator=config_generator)


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config = {
        'lookback_days': 1,
        'config': load_dict_from_yaml(file_name="binance_config.yml", folder=f"{root_path}/config"),
        'resolution': "1s",
        'study_name': "xtreet_1"
    }
    task = BacktestingTask("Backtesting", timedelta(hours=24), config)
    asyncio.run(task.execute())
