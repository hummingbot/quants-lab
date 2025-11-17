import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any, Dict

import pandas as pd

from core.data_sources import CLOBDataSource
from core.data_structures.trading_rules import TradingRules
from core.tasks import BaseTask, TaskContext

logging.getLogger("asyncio").setLevel(logging.CRITICAL)


class FundingRatesTask(BaseTask):
    """Download funding rates data from exchanges and compute rate differences."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Configuration with defaults
        task_config = self.config.config
        self.connector_names = task_config.get("connector_names", ["binance_perpetual"])
        self.quote_asset = task_config.get("quote_asset", "USDT")
        self.n_top_funding_rates_per_group = task_config.get("n_top_funding_rates_per_group", 5)
        
        # Initialize clients
        self.clob = CLOBDataSource()

    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution, including validation of prerequisites."""
        try:
            await super().setup(context)
            
            # Validate prerequisites
            if not self.connector_names:
                raise RuntimeError("connector_names not configured")
            
            logging.info(f"Setup completed for {context.task_name}")
            logging.info(f"Connectors: {self.connector_names}")
            logging.info(f"Quote asset: {self.quote_asset}")
            
        except Exception as e:
            logging.error(f"Setup failed: {e}")
            raise
    
    async def cleanup(self, context: TaskContext, result) -> None:
        """Cleanup after task execution."""
        try:
            await super().cleanup(context, result)
            logging.info(f"Cleanup completed for {context.task_name}")
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Main execution logic."""
        start_execution = datetime.now(timezone.utc)
        logging.info(f"Starting funding rates collection for {self.connector_names}")
        
        try:
            # Track statistics
            stats = {
                "connectors_processed": 0,
                "connectors_total": len(self.connector_names),
                "funding_rates_collected": 0,
                "rate_differences_computed": 0,
                "errors": 0
            }
            
            # Process each connector
            for connector_name in self.connector_names:
                try:
                    logging.info(f"Processing connector: {connector_name}")
                    current_timestamp = datetime.now(timezone.utc).timestamp()
                    
                    # Get connector and trading rules
                    connector = self.clob.get_connector(connector_name)
                    trading_rules: TradingRules = await self.clob.get_trading_rules(connector_name)
                    trading_pairs = trading_rules.get_all_trading_pairs()
                    
                    # Fetch funding rates for all pairs
                    funding_tasks = [
                        connector._orderbook_ds.get_funding_info(trading_pair) 
                        for trading_pair in trading_pairs
                    ]
                    funding_rates_response = await asyncio.gather(*funding_tasks)
                    
                    # Process funding rates data
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
                    
                    # Store funding rates
                    await self.mongodb_client.insert_documents(
                        collection_name="funding_rates",
                        documents=funding_rates,
                        index=[
                            ("trading_pair", 1),
                            ("connector_name", 1),
                            ("next_funding_utc_timestamp", 1)
                        ]
                    )
                    
                    # Compute rate differences if we have enough pairs
                    if len(funding_rates) > 1:
                        df = pd.DataFrame(funding_rates)
                        combinations_list = list(combinations(df['trading_pair'], 2))
                        
                        results = []
                        for pair1, pair2 in combinations_list:
                            rate1 = df.loc[df['trading_pair'] == pair1, 'rate'].values[0]
                            rate2 = df.loc[df['trading_pair'] == pair2, 'rate'].values[0]
                            rate_difference = rate1 - rate2
                            results.append({
                                'timestamp': current_timestamp,
                                'connector_name': connector_name,
                                'pair1': pair1,
                                'pair2': pair2,
                                'rate1': rate1,
                                'rate2': rate2,
                                'rate_difference': rate_difference,
                                'abs_rate_difference': abs(rate_difference)
                            })
                        
                        # Store processed results
                        await self.mongodb_client.insert_documents(
                            collection_name="funding_rates_processed",
                            documents=results,
                            index=[
                                ("timestamp", 1),
                                ("connector_name", 1),
                                ("pair1", 1),
                                ("pair2", 1)
                            ]
                        )
                        
                        stats["rate_differences_computed"] += len(results)
                    
                    stats["funding_rates_collected"] += len(funding_rates)
                    logging.info(f"Successfully processed {len(funding_rates)} funding rates for {connector_name}")
                    
                except Exception as e:
                    stats["errors"] += 1
                    logging.exception(f"Error processing connector {connector_name}: {e}")
                    continue
                
                stats["connectors_processed"] += 1
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_execution
            result = {
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "connectors": self.connector_names,
                "stats": stats,
                "duration_seconds": duration.total_seconds()
            }
            
            logging.info(f"Funding rates collection completed: {stats}")
            return result
            
        except Exception as e:
            logging.error(f"Error executing funding rates task: {e}")
            raise
    
    async def on_success(self, context: TaskContext, result) -> None:
        """Handle successful execution."""
        stats = result.result_data.get("stats", {})
        logging.info(f"✓ FundingRatesTask succeeded in {result.duration_seconds:.2f}s")
        logging.info(f"  - Connectors: {stats.get('connectors_processed', 0)}/{stats.get('connectors_total', 0)}")
        logging.info(f"  - Funding rates: {stats.get('funding_rates_collected', 0)}")
        logging.info(f"  - Rate differences: {stats.get('rate_differences_computed', 0)}")
        if stats.get('errors', 0) > 0:
            logging.warning(f"  - Errors: {stats.get('errors', 0)}")
    
    async def on_failure(self, context: TaskContext, result) -> None:
        """Handle failed execution."""
        logging.error(f"✗ FundingRatesTask failed: {result.error_message}")
        logging.error(f"  Execution ID: {context.execution_id}")
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """Handle retry attempt."""
        logging.warning(f"🔄 FundingRatesTask retry attempt {attempt}: {error}")


async def main():
    """Standalone execution for testing."""
    from core.tasks.base import TaskConfig, ScheduleConfig
    
    # Create v2.0 TaskConfig
    config = TaskConfig(
        name="funding_rates_task_test",
        enabled=True,
        task_class="tasks.data_collection.funding_rates_task.FundingRatesTask",
        schedule=ScheduleConfig(
            type="frequency",
            frequency_hours=1.0
        ),
        config={
            "use_mongodb": True,
            "connector_names": ["binance_perpetual"],
            "quote_asset": "USDT",
            "n_top_funding_rates_per_group": 5
        }
    )
    
    # Create and run task
    task = FundingRatesTask(config)
    result = await task.run()
    
    print(f"Task completed with status: {result.status}")
    if result.result_data:
        stats = result.result_data.get("stats", {})
        print(f"Collected {stats.get('funding_rates_collected', 0)} funding rates")
        print(f"Computed {stats.get('rate_differences_computed', 0)} rate differences")
        print(f"Processed {stats.get('connectors_processed', 0)} connectors")
    if result.error_message:
        print(f"Error: {result.error_message}")


if __name__ == "__main__":
    asyncio.run(main())
