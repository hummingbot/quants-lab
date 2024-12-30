import time
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from core.data_structures.candles import Candles
from core.services.timescale_client import TimescaleClient


class CoinGlassClient(TimescaleClient):
    @staticmethod
    def get_liquidation_aggregated_history_table_name(
        trading_pair: str, interval: str, **kwargs
    ) -> str:
        return f"coin_glass_liquidation_aggregated_history_{trading_pair.lower().replace('-', '_')}_{interval}"

    async def create_liquidation_aggregated_history(self, table_name: str):
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        long_liquidation_usd REAL NOT NULL,
                        short_liquidation_usd REAL NOT NULL,
                        create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (timestamp)
                    );
                """)

    async def delete_liquidation_aggregated_history(
        self, trading_pair: str, interval: str, timestamp: Optional[float] = None
    ):
        table_name = self.get_liquidation_aggregated_history_table_name(
            trading_pair, interval
        )
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                query = f"DELETE FROM {table_name}"
                params = []

                if timestamp is not None:
                    query += " WHERE timestamp < $1"
                    params.append(datetime.fromtimestamp(timestamp))
                await conn.execute(query, *params)

    async def append_liquidation_aggregated_history(
        self, table_name: str, data: List[Tuple[str, str, int]]
    ):
        updated_data = [(t, float(l), float(s)) for l, s, t in data]
        print(updated_data)
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                await self.create_liquidation_aggregated_history(table_name)
                await conn.executemany(
                    f"""
                        INSERT INTO {table_name} (timestamp, long_liquidation_usd, short_liquidation_usd)
                        VALUES (to_timestamp($1), $2, $3)
                        ON CONFLICT (timestamp) 
                        DO UPDATE SET 
                            long_liquidation_usd = EXCLUDED.long_liquidation_usd,
                            short_liquidation_usd = EXCLUDED.short_liquidation_usd;
                    """,
                    updated_data,
                )

    async def get_first_liquidation_aggregated_history_timestamp(
        self, trading_pair: str, interval: str
    ) -> Optional[float]:
        table_name = self.get_liquidation_aggregated_history_table_name(
            trading_pair, interval
        )
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(f"""
                    SELECT MIN(timestamp) FROM {table_name}
                """)
                return result.timestamp() if result else None

    async def get_last_liquidation_aggregated_history_timestamp(
        self, trading_pair: str, interval: str
    ) -> Optional[float]:
        table_name = self.get_liquidation_aggregated_history_table_name(
            trading_pair, interval
        )
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(f"""
                    SELECT MAX(timestamp) FROM {table_name}
                """)
                return result.timestamp() if result else None

    async def get_liquidation_aggregated_history(
        self,
        connector_name: str,
        trading_pair: str,
        interval: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> pd.DataFrame:
        candle_table_name = self.get_ohlc_table_name(
            connector_name, trading_pair, interval
        )
        liquidation_aggregated_history_table_name = (
            self.get_liquidation_aggregated_history_table_name(trading_pair, interval)
        )
        async with self.pool.acquire() as conn:
            query = f"""
                SELECT 
                    c.timestamp, 
                    c.open, 
                    c.high, 
                    c.low, 
                    c.close, 
                    c.volume, 
                    l.long_liquidation_usd,
                    l.short_liquidation_usd 
                FROM 
                    {candle_table_name} c
                JOIN 
                    {liquidation_aggregated_history_table_name} l 
                ON 
                    c.timestamp = l.timestamp
                WHERE 
                    c.timestamp BETWEEN $1 AND $2
                ORDER BY 
                    l.timestamp;
            """
            start_dt = (
                datetime.fromtimestamp(start_time) if start_time else datetime.min
            )
            end_dt = datetime.fromtimestamp(end_time) if end_time else datetime.max
            rows = await conn.fetch(query, start_dt, end_dt)
        df = pd.DataFrame(
            rows,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "long_liquidation_usd",
                "short_liquidation_usd",
            ],
        )
        df.set_index("timestamp", inplace=True)
        df = df.astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
                "long_liquidation_usd": float,
                "short_liquidation_usd": float,
            }
        )

        return df

    async def get_liquidation_aggregated_history_last_day(
        self, connector_name: str, trading_pair: str, interval: str, days: int
    ) -> Candles:
        end_time = int(time.time())
        start_time = end_time - days * 24 * 60 * 60
        return await self.get_candles(
            connector_name, trading_pair, interval, start_time, end_time
        )

    def get_global_long_short_account_ratio_table_name(
        self, trading_pair: str, interval: str, connector_name: str, **kwargs
    ) -> str:
        return f"coin_glass_global_long_short_account_ratio_{connector_name}_{trading_pair.lower().replace('-', '_')}_{interval}"

    async def create_global_long_short_account_ratio(self, table_name: str):
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TIMESTAMPTZ NOT NULL,
                        long_account REAL NOT NULL,
                        short_account REAL NOT NULL,
                        long_short_ratio REAL NOT NULL,
                        create_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (timestamp)
                    );
                """)

    async def append_global_long_short_account_ratio(
        self, table_name: str, data: List[Tuple[int, str, str, str]]
    ):
        updated_data = [(t, float(l), float(s), float(r)) for t, l, s, r in data]
        if self.pool is not None:
            async with self.pool.acquire() as conn:
                await self.create_global_long_short_account_ratio(table_name)
                await conn.executemany(
                    f"""
                        INSERT INTO {table_name} (timestamp, long_account, short_account, long_short_ratio)
                        VALUES (to_timestamp($1), $2, $3, $4)
                        ON CONFLICT (timestamp) 
                        DO UPDATE SET 
                            long_account = EXCLUDED.long_account,
                            short_account = EXCLUDED.short_account,
                            long_short_ratio = EXCLUDED.long_short_ratio;
                    """,
                    updated_data,
                )
