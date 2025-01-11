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
    from tasks.backtesting.xgridt_backtesting_task import XGridTBacktestingTask
    orchestrator = TaskOrchestrator()
    optuna_config = {
        "host": os.getenv("OPTUNA_HOST", "localhost"),
        "port": os.getenv("OPTUNA_PORT", 5433),
        "user": os.getenv("OPTUNA_USER", "admin"),
        "password": os.getenv("OPTUNA_PASSWORD", "admin"),
        "database": os.getenv("OPTUNA_DB", "optimization_database")
    }
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "connector_name": "binance_perpetual",
        "total_amount": 100,
        "lookback_days": 120,
        "end_time_buffer_hours": 6,
        "resolution": "1m",
        "selected_pairs": ['1000SHIB-USDT', 'WLD-USDT', 'ACT-USDT', '1000BONK-USDT', 'DOGE-USDT', 'AGLD-USDT',
                           'SUI-USDT', '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT',
                           'HMSTR-USDT', 'TROY-USDT', '1000PEPE-USDT', '1000X-USDT', 'PNUT-USDT', 'SOL-USDT',
                           'XRP-USDT', 'SWELL-USDT'],
        "optuna_config": optuna_config
    }

    task = XGridTBacktestingTask("Backtesting", timedelta(hours=12), config)
    orchestrator.add_task(task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
