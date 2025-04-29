from datetime import timedelta, datetime
from itertools import combinations

import pandas as pd
from dotenv import load_dotenv
import logging
import time
import os
import asyncio
from typing import Dict, Any

from core.data_structures.trading_rules import TradingRules
from core.data_sources import CLOBDataSource
from core.services.mongodb_client import MongoClient
from core.task_base import BaseTask

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class FundingRatesTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.mongo_client = MongoClient(config.get("mongo_uri", ""), database="quants_lab")
        self.clob = CLOBDataSource()

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            for connector_name in self.config.get("connector_names", ["binance_perpetual"]):
                current_timestamp = datetime.now().timestamp()
                connector = self.clob.get_connector(connector_name)
                trading_rules: TradingRules = await self.clob.get_trading_rules(connector_name)
                trading_pairs = trading_rules.get_all_trading_pairs()

                tasks = [connector._orderbook_ds.get_funding_info(trading_pair) for trading_pair in trading_pairs]
                funding_rates_response = await asyncio.gather(*tasks)
                funding_rates = []
                for funding_rate in funding_rates_response:
                    funding_rates.append({
                        "index_price": float(funding_rate.index_price),
                        "mark_price": float(funding_rate.mark_price),
                        "next_funding_utc_timestamp": funding_rate.next_funding_utc_timestamp,
                        "rate": float(funding_rate.rate),
                        "trading_pair": funding_rate.trading_pair,
                        "connector_name": connector_name,
                        "timestamp": current_timestamp
                    })

                await self.mongo_client.insert_documents(collection_name="funding_rates",
                                                         documents=funding_rates,
                                                         index=[("trading_pair", 1),
                                                                ("connector_name", 1),
                                                                ("next_funding_utc_timestamp", 1)])

                df = pd.DataFrame(funding_rates)
                # Generate all possible combinations of trading pairs
                combinations_list = list(combinations(df['trading_pair'], 2))

                # Create a DataFrame to store the combinations and rate differences
                results = []

                for pair1, pair2 in combinations_list:
                    rate1 = df.loc[df['trading_pair'] == pair1, 'rate'].values[0]
                    rate2 = df.loc[df['trading_pair'] == pair2, 'rate'].values[0]
                    rate_difference = rate1 - rate2
                    results.append({
                        'timestamp': current_timestamp,
                        'pair1': pair1,
                        'pair2': pair2,
                        'rate1': rate1,
                        'rate2': rate2,
                        'rate_difference': rate_difference
                    })
                await self.mongo_client.insert_documents(collection_name="funding_rates_processed",
                                                         documents=results,
                                                         index=[("timestamp", 1),
                                                                ("pair1", 1),
                                                                ("pair2", 1)])
                logging.info(f"Successfully added {len(funding_rates)} funding rate records")

        except Exception as e:
            logging.error(f"Error in FundingRatesTask: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()


async def main():
    mongo_uri = (
        f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME', 'admin')}:"
        f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'admin')}@"
        f"{os.getenv('MONGO_HOST', 'localhost')}:"
        f"{os.getenv('MONGO_PORT', '27017')}/"
    )
    task_config = {
        "mongo_uri": mongo_uri,
        "connector_names": ["binance_perpetual"],
        "quote_asset": "USDT",
        "n_top_funding_rates_per_group": 5
    }
    task = FundingRatesTask(name="funding_rate_task",
                            frequency=timedelta(hours=1),
                            config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
