import asyncio
import logging
import os
from datetime import timedelta, datetime
from typing import Any, Dict

import numpy as np
from dotenv import load_dotenv

from core.data_sources import CLOBDataSource
import core.machine_learning.model_config as model_configs
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
        self.selected_pairs = self.config.get("selected_pairs", [])
        self.tp_values = self.config.get("tp_values", [1])
        self.feature_iteration_trials = self.config.get("feature_iteration_trials", 1)

    @staticmethod
    def generate_external_features():
        def generate_macd():
            return [[np.random.choice([12, 24, 36]), np.random.choice([24, 36, 48]), np.random.choice([9, 12, 18])]]

        def generate_rsi():
            return [np.random.choice([14, 21, 28, 35])]

        def generate_bbands():
            return [[np.random.choice([14, 21, 30]), np.random.choice([1, 2, 3]), None]]

        def generate_relative_changes():
            return [np.random.choice([1, 5, 10, 20, 50])]

        def generate_mov_avg():
            return [np.random.choice([6, 12, 24, 48])]

        def generate_volatility():
            return [np.random.choice([20, 50, 100])]

        return {
            "close": {
                'macd': generate_macd(),
                'rsi': generate_rsi(),
                'bbands': generate_bbands(),
                'relative_changes': generate_relative_changes(),
                'mov_avg': generate_mov_avg(),
                'volatility': generate_volatility(),
                'drop': True
            },
            "volume": {
                'macd': generate_macd(),
                'rsi': generate_rsi(),
                'relative_changes': generate_relative_changes(),
                'mov_avg': generate_mov_avg(),
                'volatility': generate_volatility(),
                'drop': True
            }
        }

    async def execute(self):
        clob = CLOBDataSource()
        for trading_pair in self.selected_pairs:
            logging.info(f"{self.now()} - Getting candles for {trading_pair}")
            candles = await clob.get_candles_last_days(connector_name=self.connector_name,
                                                       trading_pair=trading_pair,
                                                       interval=self.config.get("interval"),
                                                       days=self.config.get("days"))
            logging.info(f"{self.now()} - Candles fetched for {trading_pair}")
            for tp_value in self.tp_values:
                logging.info(f"{self.now()} - Trying TP: {tp_value}")
                for i in range(1, self.feature_iteration_trials + 1):
                    logging.info(f"{self.now()} - Feature iteration {i} from {self.feature_iteration_trials}")
                    external_feat = self.generate_external_features()
                    try:
                        tba = TripleBarrierAnalyser(df=candles.data,
                                                    connector_name=self.connector_name,
                                                    trading_pair=trading_pair,
                                                    tp=tp_value,
                                                    external_feat=external_feat,
                                                    root_path=self.config.get("root_path", ""))
                        logging.info(f"{self.now()} - Preparing data")
                        features_df = tba.prepare_data(candles.data)
                        logging.info(f"{self.now()} - Doing some magic")
                        for model_config in self.config.get("model_configs"):
                            tba.transform_train(features_df=features_df, model_config=model_config)
                            tba.analyse()
                            logging.info(f"{self.now()} - Preparing next trial...")
                    except Exception as e:
                        print(f"Exception running trial: {e}")
                        continue

    @staticmethod
    def now():
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


async def main():
    config = {
        "root_path": os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        "selected_pairs": ["1000PEPE-USDT"],
        "connector_name": "binance_perpetual",
        "interval": "1m",
        "days": 7,
        "tp_values": [1, 2, 4, 6],
        "feature_iteration_trials": 5,
        "model_configs": [model_configs.RF_CONFIG]
    }

    task = TripleBarrierAnalyzerTask("Triple Barrier Analyzer", timedelta(hours=6), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
