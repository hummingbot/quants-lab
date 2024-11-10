import asyncio
import json
import logging
from typing import Dict, Any

import pandas as pd
import pandas_ta as ta
import os
from dotenv import load_dotenv
from datetime import timedelta, datetime, timezone

from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)
load_dotenv()


class MarketScreenerTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any], ts_client: TimescaleClient = None):
        super().__init__(name, frequency, config)
        self.ts_client = ts_client or TimescaleClient(os.getenv("TIMESCALE_HOST", "localhost"))
        self.intervals = self.config["intervals"]
        self.interval_mapping = {
            "1m": "one_min",
            "3m": "three_min",
            "5m": "five_min",
            "15m": "fifteen_min",
            "1h": "one_hour"
        }

    async def execute(self):
        try:
            await self.ts_client.connect()
            available_pairs = await self.ts_client.get_available_pairs()

            for connector_name, trading_pair in available_pairs:
                await self.process_pair(connector_name, trading_pair)

        except ConnectionError as e:
            logging.exception(f"{self.now()} - Database connection failed\n {e}")
        except Exception as e:
            logging.exception(f"{self.now()} - Unexpected error during execution\n {e}")

    async def process_pair(self, connector_name, trading_pair):
        """Process metrics for a single trading pair."""
        try:
            candles = await self.ts_client.get_candles(connector_name, trading_pair, interval="1h")
            screener_metrics = self.calculate_global_screener_metrics(
                candles_df=candles.data,
                connector_name=connector_name,
                trading_pair=trading_pair
            )

            interval_screener_metrics = await self.calculate_interval_metrics(connector_name, trading_pair)
            screener_metrics.update(interval_screener_metrics)
            screener_metrics = {key: json.dumps(value) if isinstance(value, dict) else value for key, value in screener_metrics.items()}

            await self.ts_client.append_screener_metrics(screener_metrics)

        except (ValueError, TypeError) as e:
            logging.exception(f"{self.now()} - Error calculating metrics for {trading_pair}\n {e}")
        except Exception as e:
            logging.exception(f"{self.now()} - Unexpected error processing pair {trading_pair}\n {e}")

    async def calculate_interval_metrics(self, connector_name, trading_pair):
        """Calculate metrics for each selected interval."""
        interval_screener_metrics = {}
        for selected_interval in self.intervals:
            mapped_interval = self.interval_mapping[selected_interval]
            try:
                candles = await self.ts_client.get_candles(connector_name, trading_pair, interval=selected_interval)
                interval_screener_metrics[mapped_interval] = self.calculate_interval_screener_metrics(candles.data)
            except Exception as e:
                logging.exception(
                    f"{self.now()} - Error processing interval {selected_interval} for {trading_pair}\n {e}")
                interval_screener_metrics[mapped_interval] = {}
        return interval_screener_metrics

    def calculate_global_screener_metrics(self, candles_df: pd.DataFrame, connector_name: str, trading_pair: str):
        df = candles_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")

        # 1. Price Analysis
        # Describe price statistics
        global_metrics = {
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "start_time": df.index.min().tz_localize("UTC"),  # or your desired timezone
            "end_time": df.index.max().tz_localize("UTC"),
            "price": df["close"].describe().to_dict()}

        # Price Change Over Periods (% CBO 24h, 1w, 1m)
        price_resampled = df['close'].resample('1d').last()  # Daily resampling for consistent periods
        global_metrics['price_cbo'] = {
            '24h': self.percent_change(price_resampled, 1),
            '1w': self.percent_change(price_resampled, 7),
            '1m': self.percent_change(price_resampled, 28)
        }

        # 2. Volume Analysis
        # 24h Volume USD
        last_24h = df.loc[df.index >= (df.index[-1] - timedelta(hours=24)), 'volume'].sum()
        global_metrics['volume_24h'] = last_24h

        # Volume % CBO for Different Periods
        volume_resampled = df['volume'].resample('1d').sum()
        global_metrics['volume_cbo'] = {
            '24h': self.percent_change(volume_resampled, 1),
            '1w': self.percent_change(volume_resampled, 7),
            '1m': self.percent_change(volume_resampled, 30)
        }
        return global_metrics

    @staticmethod
    def calculate_interval_screener_metrics(candles_df: pd.DataFrame):
        interval_metrics = {}

        df = candles_df.copy()
        df['atr_24h'] = ta.atr(df['high'], df['low'], df['close'], length=24)
        df['atr_1w'] = ta.atr(df['high'], df['low'], df['close'], length=7 * 24)
        df['natr_24h'] = ta.natr(df['high'], df['low'], df['close'], length=24)
        df['natr_1w'] = ta.natr(df['high'], df['low'], df['close'], length=7 * 24)
        interval_metrics['natr'] = df[['natr_24h', 'natr_1w']].describe().to_dict()

        # Bollinger Bands Width (50, 100, 200 / 2.0)
        for window in [50, 100, 200]:
            bb = ta.bbands(df['close'], length=window, std=2)
            interval_metrics[f'bb_width_{window}'] = bb[f'BBB_{window}_2.0'].mean() / 200
        return interval_metrics

    @staticmethod
    def percent_change(series, period):
        shifted = series.shift(period)
        if shifted.iloc[-1] == 0 or pd.isna(shifted.iloc[-1]):
            return None
        return (series.iloc[-1] - shifted.iloc[-1]) / shifted.iloc[-1]

    @staticmethod
    def now():
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')


async def main():
    config = {
        "intervals": ["1m", "3m", "5m", "15m", "1h"]
    }
    task = MarketScreenerTask("Market Screener", timedelta(hours=1), config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
