import asyncpg
import pandas as pd


class CoinGlassDataProvider:
    endpoints = {
        "liquidation_aggregated_history": "/api/futures/liquidation/v2/aggregated-history",
        "global_long_short_account_ratio": "/api/futures/globalLongShortAccountRatio/history",
        "aggregated_open_interest_history": "/api/futures/openInterest/ohlc-aggregated-history",
        "funding_rate": "/api/futures/fundingRate/ohlc-history",
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "admin",
        password: str = "admin",
        database: str = "timescaledb",
    ):
        self.db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self.pool = None

    async def connect(self):
        """Establish a connection pool to the database."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(**self.db_config)

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()

    @staticmethod
    def get_liquidation_aggregated_history_table_name(
        trading_pair: str, interval: str
    ) -> str:
        return f"coin_glass_liquidation_aggregated_history_{trading_pair.lower().replace('-', '_')}_{interval}"

    @staticmethod
    def get_aggregated_open_interest_history_table_name(
        trading_pair: str, interval: str
    ) -> str:
        return f"coin_glass_aggregated_open_interest_history_{trading_pair.lower().replace('-', '_')}_{interval}"

    @staticmethod
    def get_funding_rate_table_name(
        trading_pair: str, interval: str, connector_name: str, **kwargs
    ) -> str:
        return f"coin_glass_funding_rate_{connector_name}_{trading_pair.lower().replace('-', '_')}_{interval}"

    async def _fetch_data_from_db(
        self, table_name: str, max_records: int
    ) -> pd.DataFrame:
        """Helper method to execute the database query and return a DataFrame."""
        await self.connect()
        async with self.pool.acquire() as conn:
            query = f"""
                SELECT * FROM {table_name} 
                ORDER BY timestamp DESC
                LIMIT {max_records};
            """
            rows = await conn.fetch(query)
        await self.close()
        df = pd.DataFrame(
            rows,
        )
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    async def get_table_df(
        self,
        endpoint: str,
        trading_pair: str,
        interval: str,
        connector_name: str,
        max_records: int,
    ) -> pd.DataFrame:
        """General method to call the specific get_xxx_df function based on the endpoint."""
        table_name = self.get_table_name(
            endpoint, trading_pair, interval, connector_name
        )
        return await self._fetch_data_from_db(table_name, max_records)

    def get_table_name(
        self, endpoint: str, trading_pair: str, interval: str, connector_name: str
    ) -> str:
        """General method to get table name dynamically based on the endpoint."""
        table_name_mapping = {
            "liquidation_aggregated_history": self.get_liquidation_aggregated_history_table_name,
            "aggregated_open_interest_history": self.get_aggregated_open_interest_history_table_name,
            "funding_rate": self.get_funding_rate_table_name,
        }

        if endpoint not in table_name_mapping:
            raise ValueError(f"Endpoint '{endpoint}' not supported for table naming.")

        return table_name_mapping[endpoint](
            trading_pair=trading_pair, interval=interval, connector_name=connector_name
        )
