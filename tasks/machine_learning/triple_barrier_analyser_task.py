import asyncio
import logging
import os
from datetime import timedelta, datetime
from typing import Any, Dict

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier

from core.data_sources import CLOBDataSource
from core.machine_learning.model_config import ModelConfig
from core.machine_learning.triple_barrier_analyser import TripleBarrierAnalyser
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


class TripleBarrierAnalyzerTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.root_path = self.config.get('root_path', "")
        self.connector_name = self.config.get("connector_name")
        self.trading_pair = self.config.get("trading_pair")

    async def execute(self):
        clob = CLOBDataSource()
        logging.info(f"{self.now()} - Getting candles")
        candles = await clob.get_candles_last_days(connector_name=self.connector_name,
                                                   trading_pair=self.trading_pair,
                                                   interval=self.config.get("interval"),
                                                   days=self.config.get("days"))
        logging.info(f"{self.now()} - Candles fetched for {self.trading_pair}")
        tba = TripleBarrierAnalyser(df=candles.data,
                                    connector_name=self.connector_name,
                                    trading_pair=self.trading_pair,
                                    external_feat=self.config.get("external_features"),
                                    root_path=self.config.get("root_path", ""))
        logging.info(f"{self.now()} - Preparing data ")
        features_df = tba.prepare_data(candles.data)

        for model_config in self.config.get("model_configs"):
            tba.transform_train(features_df=features_df, model_config=model_config)
            tba.analyse()

    @staticmethod
    def now():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
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
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
