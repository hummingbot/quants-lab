import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

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

    @staticmethod
    def get_trades_table_name(connector_name: str, trading_pair: str) -> str:
        return f"{connector_name}_{trading_pair.replace('-', '_')}_trades"

    @staticmethod
    def get_ohlc_table_name(connector_name: str, trading_pair: str, interval: str) -> str:
        return f"{connector_name}_{trading_pair.lower().replace('-', '_')}_{interval}"

    @property
    def metrics_table_name(self):
        return "summary_metrics"

    @property
    def screener_table_name(self):
        return "screener_metrics"

    async def create_candles_table(self, table_name: str):
        async with self.pool.acquire() as conn:
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    open NUMERIC NOT NULL,
                    high NUMERIC NOT NULL,
                    low NUMERIC NOT NULL,
                    close NUMERIC NOT NULL,
                    volume NUMERIC NOT NULL,
                    quote_asset_volume NUMERIC NOT NULL,
                    n_trades INTEGER NOT NULL,
                    taker_buy_base_volume NUMERIC NOT NULL,
                    taker_buy_quote_volume NUMERIC NOT NULL,
                    PRIMARY KEY (timestamp)
                )
            ''')

    async def create_screener_table(self):
        async with self.pool.acquire() as conn:
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.screener_table_name} (
                    connector_name TEXT NOT NULL,
                    trading_pair TEXT NOT NULL,
                    price JSONB NOT NULL,
                    volume_24h REAL NOT NULL,
                    price_cbo JSONB NOT NULL,
                    volume_cbo JSONB NOT NULL,
                    one_min JSONB NOT NULL,
                    three_min JSONB NOT NULL,
                    five_min JSONB NOT NULL,
                    fifteen_min JSONB NOT NULL,
                    one_hour JSONB NOT NULL,
                    start_time TIMESTAMPTZ NOT NULL,
                    end_time TIMESTAMPTZ NOT NULL
                );
            ''')

    async def create_metrics_table(self):
        async with self.pool.acquire() as conn:
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.metrics_table_name} (
                    connector_name TEXT NOT NULL,
                    trading_pair TEXT NOT NULL,
                    trade_amount REAL,
                    price_avg REAL,
                    price_max REAL,
                    price_min REAL,
                    price_median REAL,
                    from_timestamp TIMESTAMPTZ NOT NULL,
                    to_timestamp TIMESTAMPTZ NOT NULL,
                    volume_usd REAL
                );
            ''')

    async def create_trades_table(self, table_name: str):
        async with self.pool.acquire() as conn:
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
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

    async def delete_trades(self, connector_name: str, trading_pair: str, timestamp: Optional[float] = None):
        table_name = self.get_trades_table_name(connector_name, trading_pair)
        async with self.pool.acquire() as conn:
            query = f"DELETE FROM {table_name}"
            params = []

            if timestamp is not None:
                query += " WHERE timestamp < $1"
                params.append(datetime.fromtimestamp(timestamp))
            await conn.execute(query, *params)

    async def delete_candles(self, connector_name: str, trading_pair: str, interval: str,
                             timestamp: Optional[float] = None):
        table_name = self.get_ohlc_table_name(connector_name, trading_pair, interval)
        async with self.pool.acquire() as conn:
            query = f"DELETE FROM {table_name}"
            params = []

            if timestamp is not None:
                query += " WHERE timestamp < $1"
                params.append(datetime.fromtimestamp(timestamp))
            await conn.execute(query, *params)

    async def append_trades(self, table_name: str, trades: List[Tuple[int, str, str, float, float, float, bool]]):
        async with self.pool.acquire() as conn:
            await self.create_trades_table(table_name)
            await conn.executemany(f'''
                INSERT INTO {table_name} (trade_id, connector_name, trading_pair, timestamp, price, volume, sell_taker)
                VALUES ($1, $2, $3, to_timestamp($4), $5, $6, $7)
                ON CONFLICT (connector_name, trading_pair, trade_id) DO NOTHING
            ''', trades)

    async def append_candles(self, table_name: str, candles: List[Tuple[float, float, float, float, float]]):
        async with self.pool.acquire() as conn:
            await self.create_candles_table(table_name)
            await conn.executemany(f'''
                INSERT INTO {table_name} (timestamp, open, high, low, close, volume, quote_asset_volume, n_trades,
                taker_buy_base_volume, taker_buy_quote_volume)
                VALUES (to_timestamp($1), $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (timestamp) DO NOTHING
            ''', candles)

    async def append_screener_metrics(self, screener_metrics: Dict[str, Any]):
        async with self.pool.acquire() as conn:
            await self.create_screener_table()
            delete_query = f"""
                        DELETE FROM {self.screener_table_name}
                        WHERE connector_name = '{screener_metrics["connector_name"]}' AND trading_pair = '{screener_metrics["trading_pair"]}';
                        """
            query = f"""
                        INSERT INTO {self.screener_table_name} (
                            connector_name,
                            trading_pair,
                            price,
                            volume_24h,
                            price_cbo,
                            volume_cbo,
                            one_min,
                            three_min,
                            five_min,
                            fifteen_min,
                            one_hour,
                            start_time,
                            end_time
                        )
                        VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                        );
                    """
            async with self.pool.acquire() as conn:
                await self.create_screener_table()
                await conn.execute(delete_query)
                await conn.execute(
                    query,
                    screener_metrics["connector_name"],
                    screener_metrics["trading_pair"],
                    screener_metrics["price"],
                    screener_metrics["volume_24h"],
                    screener_metrics["price_cbo"],
                    screener_metrics["volume_cbo"],
                    screener_metrics["one_min"],
                    screener_metrics["three_min"],
                    screener_metrics["five_min"],
                    screener_metrics["fifteen_min"],
                    screener_metrics["one_hour"],
                    screener_metrics["start_time"],
                    screener_metrics["end_time"],

                )

    async def get_last_trade_id(self, connector_name: str, trading_pair: str, table_name: str) -> int:
        async with self.pool.acquire() as conn:
            await self.create_trades_table(table_name)
            result = await conn.fetchval(f'''
                SELECT MAX(trade_id) FROM {table_name}
                WHERE connector_name = $1 AND trading_pair = $2
            ''', connector_name, trading_pair)
            return result

    async def get_last_candle_timestamp(self, connector_name: str, trading_pair: str, interval: str) -> Optional[float]:
        table_name = self.get_ohlc_table_name(connector_name, trading_pair, interval)
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(f'''
                SELECT MAX(timestamp) FROM {table_name}
            ''')
            return result.timestamp() if result else None

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def get_min_timestamp(self, table_name):
        async with self.pool.acquire() as conn:
            start_time = await conn.fetchval(f'''
                SELECT MIN(timestamp) FROM {table_name}
                ''')
            return start_time.timestamp()

    async def get_max_timestamp(self, table_name):
        async with self.pool.acquire() as conn:
            end_timestamp = await conn.fetchval(f'''
                SELECT MAX(timestamp) FROM {table_name}
                ''')
            return end_timestamp.timestamp()

    async def get_trades(self, connector_name: str, trading_pair: str, start_time: Optional[float],
                         end_time: Optional[float] = None, chunk_size: timedelta = timedelta(hours=6)) -> pd.DataFrame:
        table_name = self.get_trades_table_name(connector_name, trading_pair)
        if end_time is None:
            end_time = datetime.now().timestamp()
        if start_time is None:
            start_time = await self.get_min_timestamp(table_name)

        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)

        async def fetch_chunk(chunk_start: datetime, chunk_end: datetime) -> pd.DataFrame:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(f'''
                    SELECT trade_id, timestamp, price, volume, sell_taker
                    FROM {table_name}
                    WHERE connector_name = $1 AND trading_pair = $2
                    AND timestamp BETWEEN $3 AND $4
                    ORDER BY timestamp
                ''', connector_name, trading_pair, chunk_start, chunk_end)

            df = pd.DataFrame(rows, columns=["trade_id", 'timestamp', 'price', 'volume', 'sell_taker'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit="s")
            df["price"] = df["price"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df

        chunks = []
        current_start = start_dt
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            chunks.append(fetch_chunk(current_start, current_end))
            current_start = current_end

        results = await asyncio.gather(*chunks)

        df = pd.concat(results, ignore_index=True)
        df.set_index('timestamp', inplace=True)
        return df

    async def compute_resampled_ohlc(self, connector_name: str, trading_pair: str, interval: str):
        candles = await self.get_candles(connector_name, trading_pair, interval, from_trades=True)
        ohlc_table_name = self.get_ohlc_table_name(connector_name, trading_pair, interval)
        async with self.pool.acquire() as conn:
            # Drop the existing OHLC table if it exists
            await conn.execute(f'DROP TABLE IF EXISTS {ohlc_table_name}')
            # Create a new OHLC table
            await conn.execute(f'''
                CREATE TABLE {ohlc_table_name} (
                    timestamp TIMESTAMPTZ NOT NULL,
                    open NUMERIC NOT NULL,
                    high NUMERIC NOT NULL,
                    low NUMERIC NOT NULL,
                    close NUMERIC NOT NULL,
                    volume NUMERIC NOT NULL,
                    PRIMARY KEY (timestamp)
                )
            ''')
            # Insert the resampled candles into the new table
            await conn.executemany(f'''
                INSERT INTO {ohlc_table_name} (timestamp, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6)
            ''', [
                (
                    datetime.fromtimestamp(row["timestamp"]),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume']
                )
                for i, row in candles.data.iterrows()
            ])

    async def execute_query(self, query: str):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query)

    def metrics_query_str(self, connector_name, trading_pair):
        table_name = self.get_trades_table_name(connector_name, trading_pair)
        return f'''
            SELECT COUNT(*) AS trade_amount,
                   AVG(price) AS price_avg,
                   MAX(price) AS price_max,
                   MIN(price) AS price_min,
                   PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS price_median,
                   MIN(timestamp) AS from_timestamp,
                   MAX(timestamp) AS to_timestamp,
                   SUM(price * volume) AS volume_usd
            FROM {table_name}
        '''

    async def get_screener_df(self):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"""
            SELECT *
            FROM {self.screener_table_name}""")
        df_cols = [
            "connector_name",
            "trading_pair",
            "price",
            "volume_24h",
            "price_cbo",
            "volume_cbo",
            "one_min",
            "three_min",
            "five_min",
            "fifteen_min",
            "one_hour",
            "start_time",
            "end_time"
        ]
        df = pd.DataFrame(rows, columns=df_cols)
        return df

    async def get_db_status_df(self):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
            SELECT *
            FROM summary_metrics""")
        df_cols = [
            "connector_name",
            "trading_pair",
            "trade_amount",
            "price_avg",
            "price_max",
            "price_min",
            "price_median",
            "from_timestamp",
            "to_timestamp",
            "volume_usd"
        ]
        df = pd.DataFrame(rows, columns=df_cols)
        return df

    async def append_db_status_metrics(self, connector_name: str, trading_pair: str):
        async with self.pool.acquire() as conn:
            query = self.metrics_query_str(connector_name, trading_pair)
            metrics = await self.execute_query(query)
            metric_data = dict(metrics[0])
            metric_data["connector_name"] = connector_name
            metric_data["trading_pair"] = trading_pair
            delete_query = f"""
                DELETE FROM {self.metrics_table_name}
                WHERE connector_name = '{metric_data["connector_name"]}' AND trading_pair = '{metric_data["trading_pair"]}';
                """
            query = f"""
                INSERT INTO {self.metrics_table_name} (
                    connector_name,
                    trading_pair,
                    trade_amount,
                    price_avg,
                    price_max,
                    price_min,
                    price_median,
                    from_timestamp,
                    to_timestamp,
                    volume_usd
                )
                VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10
                );
            """
            async with self.pool.acquire() as conn:
                await self.create_metrics_table()
                await conn.execute(delete_query)
                await conn.execute(
                    query,
                    metric_data['connector_name'],
                    metric_data['trading_pair'],
                    metric_data['trade_amount'],
                    metric_data['price_avg'],
                    metric_data['price_max'],
                    metric_data['price_min'],
                    metric_data['price_median'],
                    metric_data['from_timestamp'],
                    metric_data['to_timestamp'],
                    metric_data['volume_usd']
                )

    async def get_candles(self, connector_name: str, trading_pair: str, interval: str,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None, from_trades: bool = False) -> Candles:
        if from_trades:
            trades = await self.get_trades(connector_name=connector_name,
                                           trading_pair=trading_pair,
                                           start_time=start_time,
                                           end_time=end_time)
            if trades.empty:
                candles_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'timestamp'])
            else:
                pandas_interval = self.convert_interval_to_pandas_freq(interval)
                candles_df = trades.resample(pandas_interval).agg({"price": "ohlc", "volume": "sum"}).ffill()
                candles_df.columns = candles_df.columns.droplevel(0)
                candles_df["timestamp"] = pd.to_numeric(candles_df.index) // 1e9
        else:
            table_name = self.get_ohlc_table_name(connector_name, trading_pair, interval)
            async with self.pool.acquire() as conn:
                query = f'''
                    SELECT timestamp, open, high, low, close, volume
                    FROM {table_name}
                    WHERE timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp
                '''
                start_dt = datetime.fromtimestamp(start_time) if start_time else datetime.min
                end_dt = datetime.fromtimestamp(end_time) if end_time else datetime.max
                rows = await conn.fetch(query, start_dt, end_dt)
            candles_df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # candles_df.set_index('timestamp', inplace=True)
            candles_df['timestamp'] = candles_df['timestamp'].apply(lambda x: x.timestamp())
            candles_df = candles_df.astype(
                {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

        return Candles(candles_df=candles_df, connector_name=connector_name, trading_pair=trading_pair,
                       interval=interval)

    async def get_candles_last_days(self,
                                    connector_name: str,
                                    trading_pair: str,
                                    interval: str,
                                    days: int) -> Candles:
        end_time = int(time.time())
        start_time = end_time - days * 24 * 60 * 60
        return await self.get_candles(connector_name, trading_pair, interval, start_time, end_time)

    async def get_available_pairs(self) -> List[Tuple[str, str]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE '%_trades'
                ORDER BY table_name
            ''')

        available_pairs = []
        for row in rows:
            table_name = row['table_name']
            parts = table_name.split('_')
            base = parts[-3].upper()
            quote = parts[-2].upper()
            trading_pair = f"{base}-{quote}"
            connector_name = parts[:-3]
            if len(connector_name) > 1:
                connector_name = '_'.join(connector_name)
            available_pairs.append((connector_name, trading_pair))

        return available_pairs

    async def get_available_candles(self) -> List[Tuple[str, str, str]]:
        async with self.pool.acquire() as conn:
            # TODO: fix regex to match intervals
            timeframe_regex = r'_(\d+[smhdw])'
            rows = await conn.fetch('''
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name ~ $1
                ORDER BY table_name
            ''', timeframe_regex)
        available_candles = []
        for row in rows:
            table_name = row['table_name']
            parts = table_name.split('_')
            connector_name = parts[:-3]
            base = parts[-3].upper()
            quote = parts[-2].upper()
            trading_pair = f"{base}-{quote}"
            interval = parts[-1]
            if interval == "trades":
                continue
            if len(connector_name) > 1:
                connector_name = '_'.join(connector_name)
            available_candles.append((connector_name, trading_pair, interval))
        return available_candles

    async def get_all_candles(self, connector_name: str, trading_pair: str, interval: str) -> Candles:
        table_name = self.get_ohlc_table_name(connector_name, trading_pair, interval)
        query = f'''
            SELECT * FROM {table_name}
        '''
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
        return Candles(
            candles_df=pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_asset_volume",
                                        "n_trades", "taker_buy_base_volume", "taker_buy_quote_volume"]),
            connector_name=connector_name,
            trading_pair=trading_pair,
            interval=interval)

    async def get_data_range(self, connector_name: str, trading_pair: str) -> Dict[str, Union[datetime, str]]:
        if not connector_name or not trading_pair:
            return {"error": "Both connector_name and trading_pair must be provided"}

        table_name = self.get_trades_table_name(connector_name, trading_pair)

        query = f'''
        SELECT
        MIN(timestamp) as start_time,
        MAX(timestamp) as end_time
        FROM {table_name}
        '''

        async with self.pool.acquire() as conn:
            try:
                row = await conn.fetchrow(query)
            except asyncpg.UndefinedTableError:
                return {"error": f"Table for {connector_name} and {trading_pair} does not exist"}

        if row['start_time'] is None or row['end_time'] is None:
            return {"error": f"No data found for {connector_name} and {trading_pair}"}

        return {
            'start_time': row['start_time'],
            'end_time': row['end_time']
        }

    async def get_all_data_ranges(self) -> Dict[Tuple[str, str], Dict[str, datetime]]:
        available_pairs = await self.get_available_pairs()
        data_ranges = {}
        for connector_name, trading_pair in available_pairs:
            data_ranges[(connector_name, trading_pair)] = await self.get_data_range(connector_name, trading_pair)
        return data_ranges

    @staticmethod
    def convert_interval_to_pandas_freq(interval: str) -> str:
        """
        Converts a candle interval string to a pandas frequency string.
        """
        return INTERVAL_MAPPING.get(interval, 'T')

    async def store_grid_parameters(self, grid_params: Dict):
        """Store grid parameters in the database"""
        query = """
        INSERT INTO grid_parameters 
        (pair1, pair2, start, end, limit, side, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, NOW())
        """
        await self.pool.execute(
            query,
            grid_params["pair1"],
            grid_params["pair2"],
            grid_params["start"],
            grid_params["end"],
            grid_params["limit"],
            grid_params["side"]
        )
