import asyncio
import logging
import os
import sys
from datetime import timedelta

from sklearn.ensemble import RandomForestClassifier


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.machine_learning.triple_barrier_analyser_task import TripleBarrierAnalyzerTask
    from core.machine_learning.model_config import ModelConfig

    orchestrator = TaskOrchestrator()

    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
        "trading_pair": "1000PEPE-USDT",
        "connector_name": "binance_perpetual",
        "interval": "1m",
        "days": 7,
        "external_features": {
            "close": {
                'macd': [[12, 24, 9]]
            }
        },
        "model_configs": [
            ModelConfig(
                name="Random Forest",
                params={
                    'n_estimators': [1000],  # Or use [int(x) for x in np.linspace(start=100, stop=1000, num=3)]
                    'max_features': ['sqrt'],  # Or ['log2']
                    'max_depth': [20, 55, 100],  # Or use [int(x) for x in np.linspace(10, 100, num=3)]
                    'min_samples_split': [50, 100],
                    'min_samples_leaf': [30, 50],
                    'bootstrap': [True],
                    'class_weight': ['balanced']
                },
                model_instance=RandomForestClassifier(),
                one_vs_rest=True,
                n_iter=10,
                cv=2,
                verbose=10
            )
        ]
    }

    task = TripleBarrierAnalyzerTask("Triple Barrier Analyzer", timedelta(hours=24), config)

    orchestrator.add_task(task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
