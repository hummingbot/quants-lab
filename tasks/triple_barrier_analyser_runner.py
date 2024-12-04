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
    from tasks.machine_learning.triple_barrier_analyser_task import TripleBarrierAnalyzerTask
    import core.machine_learning.model_config as model_configs

    orchestrator = TaskOrchestrator()

    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "selected_pairs":  [
            '1000SHIB-USDT', 'WLD-USDT', 'ACT-USDT', '1000BONK-USDT', 'DOGE-USDT', 'AGLD-USDT', 'SUI-USDT',
            '1000SATS-USDT', 'MOODENG-USDT', 'NEIRO-USDT', 'HBAR-USDT', 'ENA-USDT', 'HMSTR-USDT', 'TROY-USDT',
            '1000PEPE-USDT', '1000X-USDT', 'PNUT-USDT', 'SOL-USDT', 'XRP-USDT', 'SWELL-USDT', 'BTC-USDT'
        ],
        "connector_name": "binance_perpetual",
        "interval": "1m",
        "days": 1,
        "feature_iteration_trials": 5,
        "model_configs": [model_configs.RF_CONFIG]
    }

    task = TripleBarrierAnalyzerTask("Triple Barrier Analyzer", timedelta(hours=6), config)

    orchestrator.add_task(task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
