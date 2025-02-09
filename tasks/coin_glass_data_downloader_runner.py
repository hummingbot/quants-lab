import asyncio
import logging
import os
import sys
from datetime import timedelta

from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.data_collection.coin_glass_data_downloader_task import CoinGlassDataDownloaderTask

    orchestrator = TaskOrchestrator()

    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb"),
    }

    open_interest_downloader_task = CoinGlassDataDownloaderTask(
        name="CoinGlass open interest aggregated history",
        config={
            "timescale_config": timescale_config,
            "end_point": "aggregated_open_interest_history",
            "connector_name": "binance_perpetual",
            "days_data_retention": 7,
            "api_key": os.getenv("CG_API_KEY"),
            "trading_pairs": [
                "BTC-USDT",
                "ETH-USDT",
                "1000PEPE-USDT",
                "SOL-USDT",
            ],
            "interval": ["30m","1h"],
            "limit": 1000,
        },
        frequency=timedelta(minutes=30),
    )
    liquidation_downloader_task = CoinGlassDataDownloaderTask(
        name="CoinGlass liquidation aggregated history",
        config={
            "timescale_config": timescale_config,
            "end_point": "liquidation_aggregated_history",
            "connector_name": "binance_perpetual",
            "days_data_retention": 7,
            "api_key": os.getenv("CG_API_KEY"),
            "trading_pairs": [
                "BTC-USDT",
                "ETH-USDT",
                "1000PEPE-USDT",
                "SOL-USDT",
            ],
            "interval": ["1h"],
            "limit": 1000,
        },
        frequency=timedelta(minutes=30),
    )
    long_short_ratio_downloader_task = CoinGlassDataDownloaderTask(
        name="CoinGlass long short global account ratio",
        config={
            "timescale_config": timescale_config,
            "end_point": "global_long_short_account_ratio",
            "connector_name": "binance_perpetual",
            "days_data_retention": 7,
            "api_key": os.getenv("CG_API_KEY"),
            "trading_pairs": [
                "BTC-USDT",
                "ETH-USDT",
                "1000PEPE-USDT",
                "SOL-USDT",
            ],
            "interval": ["1h"],
            "limit": 1000,
        },
        frequency=timedelta(minutes=30),
    )
    funding_rate_dowloader_task = CoinGlassDataDownloaderTask(
        name="CoinGlass funding rate",
        config={
            "timescale_config": timescale_config,
            "end_point": "funding_rate",
            "connector_name": "binance_perpetual",
            "days_data_retention": 7,
            "api_key": os.getenv("CG_API_KEY"),
            "trading_pairs": [
                "BTC-USDT",
                "ETH-USDT",
                "1000PEPE-USDT",
                "SOL-USDT",
            ],
            "interval": ["1h"],
            "limit": 1000,
        },
        frequency=timedelta(minutes=30),
    )
    funding_rate_oi_dowloader_task = CoinGlassDataDownloaderTask(
        name="CoinGlass funding rate oi",
        config={
            "timescale_config": timescale_config,
            "end_point": "funding_rate_oi",
            "connector_name": "binance_perpetual",
            "days_data_retention": 7,
            "api_key": os.getenv("CG_API_KEY"),
            "trading_pairs": [
                "BTC-USDT",
                "ETH-USDT",
                "1000PEPE-USDT",
                "SOL-USDT",
            ],
            "interval": ["1h"],
            "limit": 1000,
        },
        frequency=timedelta(minutes=30),
    )
    funding_rate_vol_dowloader_task = CoinGlassDataDownloaderTask(
        name="CoinGlass funding rate vol",
        config={
            "timescale_config": timescale_config,
            "end_point": "funding_rate_vol",
            "connector_name": "binance_perpetual",
            "days_data_retention": 7,
            "api_key": os.getenv("CG_API_KEY"),
            "trading_pairs": [
                "BTC-USDT",
                "ETH-USDT",
                "1000PEPE-USDT",
                "SOL-USDT",
            ],
            "interval": ["1h"],
            "limit": 1000,
        },
        frequency=timedelta(minutes=30),
    )

    orchestrator.add_task(open_interest_downloader_task)
    orchestrator.add_task(liquidation_downloader_task)
    orchestrator.add_task(long_short_ratio_downloader_task)
    orchestrator.add_task(funding_rate_dowloader_task)
    orchestrator.add_task(funding_rate_oi_dowloader_task)
    orchestrator.add_task(funding_rate_vol_dowloader_task)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
