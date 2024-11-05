import asyncio
import logging
import os
import sys
from datetime import timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from core.utils import load_dict_from_yaml
    from tasks.backtesting.xtreet_backtesting_task import BacktestingTask
    orchestrator = TaskOrchestrator()

    backtesting_config = {
        'config': load_dict_from_yaml(file_name="binance_config.yml", folder=f"{project_root}/tasks/config"),
        'resolution': "1s",
        'root_path': project_root
    }
    backtesting_task = BacktestingTask("Backtesting", timedelta(hours=12), backtesting_config)
    orchestrator.add_task(backtesting_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
