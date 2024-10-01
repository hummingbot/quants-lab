import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import asyncpg
import pandas as pd

from core.data_structures.candles import Candles

INTERVAL_MAPPING = {
    '1s': 's',  # seconds
    '1m': 'T',  # minutes
    '3m': '3T',
    '5m': '5T',
    '15m': '15T',
    '30m': '30T',
    '1h': 'H',  # hours
    '2h': '2H',
    '4h': '4H',
    '6h': '6H',
    '12h': '12H',
    '1d': 'D',  # days
    '3d': '3D',
    '1w': 'W'  # weeks
}


class TimescaleClient:
    def __init__(self, host: str = "localhost", port: int = 5432,
                 user: str = "admin", password: str = "admin", database: str = "timescaledb"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    async def create_trades_table(self):
        async with self.pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS Trades (
                    id SERIAL PRIMARY KEY,
                    trade_id BIGINT NOT NULL,
                    connector_name TEXT NOT NULL,
                    trading_pair TEXT NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    price NUMERIC NOT NULL,
                    volume NUMERIC NOT NULL,
                    sell_taker BOOLEAN NOT NULL,
                    UNIQUE (connector_name, trading_pair, trade_id)
                );
            ''')

    async def drop_trades_table(self):
        async with self.pool.acquire() as conn:
            await conn.execute('DROP TABLE IF EXISTS Trades')

    async def delete_trades(self, connector_name: str = None, trading_pair: str = None):
        async with self.pool.acquire() as conn:
            if connector_name and trading_pair:
                await conn.execute('''
                    DELETE FROM Trades
                    WHERE connector_name = $1 AND trading_pair = $2
                ''', connector_name, trading_pair)
            elif connector_name:
                await conn.execute('DELETE FROM Trades WHERE connector_name = $1', connector_name)
            elif trading_pair:
                await conn.execute('DELETE FROM Trades WHERE trading_pair = $1', trading_pair)
            else:
                await conn.execute('DELETE FROM Trades')

    async def append_trades(self, trades: List[Tuple[int, str, str, float, float, float, bool]]):
        async with self.pool.acquire() as conn:
            await conn.executemany('''
                INSERT INTO Trades (trade_id, connector_name, trading_pair, timestamp, price, volume, sell_taker)
                VALUES ($1, $2, $3, to_timestamp($4), $5, $6, $7)
                ON CONFLICT (connector_name, trading_pair, trade_id) DO NOTHING
            ''', trades)

    async def get_last_trade_id(self, connector_name: str, trading_pair: str) -> int:
        async with self.pool.acquire() as conn:
            result = await conn.fetchval('''
                SELECT MAX(trade_id) FROM Trades
                WHERE connector_name = $1 AND trading_pair = $2
            ''', connector_name, trading_pair)
            return result

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def get_trades(self, connector_name: str, trading_pair: str, start_time: float,
                         end_time: Optional[float] = None) -> pd.DataFrame:
        if end_time is None:
            end_time = datetime.now().timestamp()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
SELECT trade_id, timestamp, price, volume, sell_taker
FROM Trades
WHERE connector_name = $1 AND trading_pair = $2
AND timestamp BETWEEN to_timestamp($3) AND to_timestamp($4)
ORDER BY timestamp
''', connector_name, trading_pair, start_time, end_time)
        df = pd.DataFrame(rows, columns=["trade_id", 'timestamp', 'price', 'volume', 'sell_taker'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s")
        df["price"] = df["price"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df.set_index('timestamp', inplace=True)
        return df

    async def get_candles(self, connector_name: str, trading_pair: str, start_time: float, interval: str,
                          end_time: Optional[float] = None) -> Candles:
        trades = await self.get_trades(connector_name, trading_pair, start_time, end_time)
        if trades.empty:
            candles_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
        else:
            pandas_interval = self.convert_interval_to_pandas_freq(interval)
            candles_df = trades.resample(pandas_interval).agg({"price": "ohlc", "volume": "sum"}).ffill()
            candles_df.columns = candles_df.columns.droplevel(0)
            candles_df["timestamp"] = pd.to_numeric(candles_df.index) // 1e9
        return Candles(candles_df=candles_df, connector_name=connector_name, trading_pair=trading_pair, interval=interval)

    async def get_candles_last_days(self,
                                    connector_name: str,
                                    trading_pair: str,
                                    interval: str,
                                    days: int) -> Candles:
        end_time = int(time.time())
        start_time = end_time - days * 24 * 60 * 60
        return await self.get_candles(connector_name, trading_pair, start_time, interval, end_time)

    async def get_available_pairs(self) -> List[Tuple[str, str]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT DISTINCT connector_name, trading_pair
                FROM Trades
                ORDER BY connector_name, trading_pair
            ''')
        return [(row['connector_name'], row['trading_pair']) for row in rows]

    async def get_data_range(self, connector_name: Optional[str] = None,
                             trading_pair: Optional[str] = None) -> Dict[str, datetime]:
        query = '''
SELECT
MIN(timestamp) as start_time,
MAX(timestamp) as end_time
FROM Trades
'''
        params = []
        if connector_name or trading_pair:
            query += ' WHERE '
            conditions = []
            if connector_name:
                conditions.append('connector_name = $1')
                params.append(connector_name)
            if trading_pair:
                conditions.append(f'trading_pair = ${len(params) + 1}')
                params.append(trading_pair)
            query += ' AND '.join(conditions)

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)

        return {
            'start_time': row['start_time'],
            'end_time': row['end_time']
        }

    async def get_all_data_ranges(self) -> Dict[Tuple[str, str], Dict[str, datetime]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
SELECT
connector_name,
trading_pair,
MIN(timestamp) as start_time,
MAX(timestamp) as end_time
FROM Trades
GROUP BY connector_name, trading_pair
ORDER BY connector_name, trading_pair
''')
        return {
            (row['connector_name'], row['trading_pair']): {
                'start_time': row['start_time'],
                'end_time': row['end_time']
            }
            for row in rows
        }

    @staticmethod
    def convert_interval_to_pandas_freq(interval: str) -> str:
        """
        Converts a candle interval string to a pandas frequency string.
        """
        return INTERVAL_MAPPING.get(interval, 'T')
