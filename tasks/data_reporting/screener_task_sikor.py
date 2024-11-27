import asyncio
import logging
import os
import time
from typing import Any, Dict
from datetime import timedelta

import pandas as pd
from dotenv import load_dotenv

import numpy as np

from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

load_dotenv()
logging.basicConfig(level=logging.INFO)


# Base class for common functionalities like database connection and email sending
class ScreenerSikorTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.ts_client = TimescaleClient(host=self.config.get("host", "localhost"))

    @staticmethod
    def get_volatility(df, window):
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=window).std() * np.sqrt(window)
        return df['volatility'].iloc[-1]

    @staticmethod
    def get_volume_imbalance(df, window):
        # Calculate volume metrics
        df["volume_usd"] = df["volume"] * df["close"]
        df["buy_taker_volume_usd"] = df["taker_buy_base_volume"] * df["close"]
        df["sell_taker_volume_usd"] = df["volume_usd"] - df["buy_taker_volume_usd"]
        # Calculate buy/sell imbalance
        df["buy_sell_imbalance"] = df["buy_taker_volume_usd"] - df["sell_taker_volume_usd"]
        # Calculate rolling total volume
        rolling_total_volume_usd = df["volume_usd"].rolling(window=window, min_periods=1).sum()
        return rolling_total_volume_usd.iloc[-1]

    async def execute(self):
        await self.ts_client.connect()
        available_candles = await self.ts_client.get_available_candles()
        filtered_candles = [candle for candle in available_candles
                            if candle[2] == self.config["interval"] and candle[0] == self.config["connector_name"]]
        candles_tasks = [self.ts_client.get_all_candles(candle[0], candle[1], candle[2]) for candle in filtered_candles]
        candles = await asyncio.gather(*candles_tasks)
        report = []
        for candle in candles:
            trading_pair = candle.trading_pair
            df = candle.data
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["taker_buy_base_volume"] = df["taker_buy_base_volume"].astype(float)
            # Calculate logarithmic returns
            volatility = self.get_volatility(df, self.config.get("volatility_window", 50))
            volume_imbalance = self.get_volume_imbalance(df, self.config.get("volume_window", 50))
            # ADD METHOD FOR ORDER BOOK IMBALANCE
            report.append({
                "trading_pair": trading_pair,
                "volatility": volatility,
                "volume_imbalance": volume_imbalance
            })
        logging.info(pd.DataFrame(report))

async def main():
    config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "connector_name": "binance_perpetual",
        "interval": "15m",
        "days": 30,
        "volatility_window": 50,
        "volume_window": 50,
    }
    task = ScreenerSikorTask(name="Report Generator", frequency=timedelta(hours=12), config=config)
    times_before_start = time.time()
    await task.execute()
    times_after_start = time.time()
    duration_seconds = times_after_start - times_before_start
    print(f"Task took {duration_seconds} seconds to complete")


if __name__ == "__main__":
    asyncio.run(main())
