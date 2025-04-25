import time
import warnings
from datetime import timedelta
from itertools import combinations

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logging
import asyncio
import os
from typing import List, Dict, Any, Tuple

from statsmodels.tsa.stattools import coint, grangercausalitytests
from pyinform.transferentropy import transfer_entropy
from dtaidistance import dtw

from core.data_sources import CLOBDataSource
from core.data_structures.candles import Candles
from core.services.mongodb_client import MongoClient
from core.task_base import BaseTask

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("core.data_sources.clob").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore', category=FutureWarning)
load_dotenv()


class CointegrationV2Task(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.mongo_client = MongoClient(self.config.get("mongo_uri"))
        self.root_path = "../../.."
        self.clob = CLOBDataSource()

    async def initialize(self):
        """Initialize connections and resources."""
        self.reset_metadata()
        await self.mongo_client.connect()

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            self.clob = CLOBDataSource()  # we need to refresh because it breaks on the second iteration
            candles = await self.get_candles()
            trading_pairs = self.get_trading_pairs_filtered_by_volume(candles)
            self.logs.append(f"{self.now()} - Filtered {len(trading_pairs)} trading pairs out of {len(candles)}, "
                             f"starting {len(trading_pairs) ** 2 - len(trading_pairs)} cointegration analysis...")
            candles_dict = {
                candle.trading_pair: candle
                for candle in candles
                if candle.trading_pair in trading_pairs
            }
            results = self.analyze_all_pairs(trading_pairs, candles_dict)
            results_df = pd.DataFrame(results)

            cointegration_results = []
            processed_pairs = set()
            now = time.time()
            for _, row in results_df.iterrows():
                pair_key = tuple(sorted([row['base'], row['quote']]))

                # Skip if we've already processed this pair
                if pair_key in processed_pairs:
                    continue

                # Get both directions for this pair (2-rows df)
                pair_analysis = results_df[
                    ((results_df['base'] == row['base']) & (results_df['quote'] == row['quote'])) |
                    ((results_df['base'] == row['quote']) & (results_df['quote'] == row['base']))
                    ]

                # Check if both directions are cointegrated
                if not (pair_analysis['p_value'] < self.config.get("p_value_threshold", 0.05)).all():
                    continue

                # Calculate average cointegration value
                coint_value = pair_analysis['p_value'].mean()
                lookback_days_timestamp = pair_analysis['lookback_days_timestamp'].min()

                pair_analysis_a = pair_analysis.iloc[0]
                pair_analysis_b = pair_analysis.iloc[1]

                # Create a result dictionary for this pair
                result = {
                    'direction_a': {
                        "dominant": pair_analysis_a["base"],
                        "hedge": pair_analysis_a["quote"],
                        "p_value": pair_analysis_a["p_value"],
                        "dominance": pair_analysis_a["dominance"]
                    },
                    'direction_b': {
                        "dominant": pair_analysis_b["base"],
                        "hedge": pair_analysis_b["quote"],
                        "p_value": pair_analysis_b["p_value"],
                        "dominance": pair_analysis_b["dominance"],
                    },
                    'coint_value': coint_value,
                    'connector_name': self.config["candles_config"]["connector_name"],
                    'interval': self.config["candles_config"]["interval"],
                    'lookback_days': self.config["lookback_days"],
                    'lookback_days_timestamp': lookback_days_timestamp,
                    'timestamp': now,
                    'metadata': self.metadata
                }

                cointegration_results.append(result)

                # Mark this pair as processed
                processed_pairs.add(pair_key)

            # Sort results by cointegration value
            cointegration_results.sort(key=lambda x: x['coint_value'])

            # Print summary
            self.logs.append(f"{self.now()} - Found {len(cointegration_results)} cointegrated pairs.")

            if len(cointegration_results) > 0:
                await self.mongo_client.insert_documents(collection_name="cointegration_results_v2",
                                                         db_name="quants_lab",
                                                         documents=cointegration_results,
                                                         index=[("base", 1), ("quote", 1)])
                logging_msg = f"{self.now()} - Successfully added {len(cointegration_results)} cointegration records"
                logging.info(logging_msg)
                self.logs.append(logging_msg)
        except Exception as e:
            logging.error(f"Error in Cointegration Task: {str(e)}")
            raise

    async def get_candles(self):
        trading_rules = await self.clob.get_trading_rules(connector_name=self.config["connector_name"])
        trading_pairs = trading_rules.filter_by_quote_asset("USDT").get_all_trading_pairs()
        candles_config = self.config.get("candles_config", {})
        if self.config.get("update_candles"):
            try:
                candles = await self.clob.get_candles_batch_last_days(trading_pairs=trading_pairs, **candles_config)
                self.clob.dump_candles_cache(self.root_path)
            except Exception as e:
                candles = None
                print(e)
        else:
            self.clob.load_candles_cache(self.root_path)
            candles = [self.clob.get_candles_from_cache(*key) for key, _ in self.clob.candles_cache.items()]
        return candles

    def get_trading_pairs_filtered_by_volume(self, candles: List[Candles]):
        all_candles = pd.DataFrame()
        for candle in candles:
            df = candle.data.copy()
            df["trading_pair"] = candle.trading_pair
            all_candles = pd.concat([all_candles, df])
        grouped_candles = all_candles.groupby("trading_pair")["quote_asset_volume"].sum().reset_index()
        volume_filter_quantile = grouped_candles["quote_asset_volume"].quantile(
            self.config.get("volume_quantile", 0.75))
        selected_candles = grouped_candles[grouped_candles["quote_asset_volume"] >= volume_filter_quantile]
        trading_pairs = [
            candle.trading_pair for candle in candles
            if candle.trading_pair in selected_candles.trading_pair.values
        ]
        return trading_pairs

    def analyze_all_pairs(self, trading_pairs, candles_dict):
        results = []
        for pair1, pair2 in combinations(trading_pairs, 2):
            try:
                candle1 = candles_dict[pair1]
                candle2 = candles_dict[pair2]
                interval = candle1.interval

                for cut_value in range(0, self.config["max_lookback_steps"], self.config["lookback_step"]):
                    close1 = candle1.data.close.iloc[cut_value:].pct_change().add(1).cumprod().dropna()
                    close2 = candle2.data.close.iloc[cut_value:].pct_change().add(1).cumprod().dropna()
                    cross_corr = self.cross_correlation_function(pair1, pair2, close1, close2, max_lag=6)
                    # dtw_dist = self.dtw_distance_analysis(pair1, pair2, close1, close2)
                    dtw_dist = {}

                    # Analyze both directions
                    result1 = self._analyze_pair(pair1, pair2, close1, close2, cross_corr, dtw_dist, interval)
                    result2 = self._analyze_pair(pair2, pair1, close2, close1, cross_corr, dtw_dist, interval)
                    results.extend([result1, result2])
            except Exception as e:
                print(f"Error analyzing {pair1} vs {pair2}: {str(e)}")
                continue
        return results

    def _analyze_pair(self, pair1, pair2, close1, close2, cross_corr, dtw_dist, interval: str = "15m"):
        try:
            # One-direction dominance
            granger = self.granger_causality(pair1, pair2, close1, close2, max_lag=6)
            entropy = self.transfer_entropy_analysis(pair1, pair2, close1, close2, k=1, bins=3)

            cointegration = self.analyze_pair_cointegration(close1, close2, interval)

            return {
                'base': pair1,
                'quote': pair2,
                'p_value': cointegration['p_value'],
                'lookback_days_timestamp': cointegration['lookback_days_timestamp'],
                'dominance': {
                    'cross_correlation': cross_corr,
                    'granger_causality': granger,
                    'dtw_distance': dtw_dist,
                    'entropy_transfer': entropy,
                }
            }
        except Exception:
            raise

    @staticmethod
    def cross_correlation_function(pair1, pair2, close1, close2, max_lag=6):
        """
        Computes the lead-lag relationship between two trading pairs based on a certain interval percentage returns.

        This method analyzes which market tends to move first (dominant) and which tends to follow (follower),
        by computing the Pearson correlation between shifted return series across a range of time lags.

        Parameters:
        -----------
        candles1 : Candles
        candles2 : Candles
        max_lag : int, default=6
            Maximum number of lags (in both directions) to test. A lag of ±6 implies up to 30 minutes
            forward/backward testing when using 5-minute candles.

        Returns:
        --------
        results : dict
            A dictionary containing:
            - "best_lag": The lag with the highest correlation.
            - "best_lag_corr": The maximum correlation value.
            - "dominant": The trading pair that tends to move first (None if lag is 0).
            - "follower": The trading pair that tends to follow (None if lag is 0).

        Notes:
        ------
        - A positive lag means `candles1` leads `candles2`.
        - A negative lag means `candles2` leads `candles1`.
        - If the best lag is 0, both series are considered synchronous and no leader is assigned.
        """
        results = {
            "dominant": None,
            "follower": None
        }

        # Align
        min_len = min(len(close1), len(close2))
        series1 = close1[-min_len:]
        series2 = close2[-min_len:]

        lags = np.arange(-max_lag, max_lag + 1)
        correlations = [series1.corr(series2.shift(lag)) for lag in lags]

        max_corr = max(correlations)
        best_lag = lags[np.argmax(correlations)]

        results["best_lag"] = int(best_lag)
        results["best_lag_corr"] = float(max_corr)

        if best_lag == 0:
            return results

        results["dominant"] = pair2 if best_lag < 0 else pair1
        results["follower"] = pair1 if best_lag < 0 else pair2
        return results

    @staticmethod
    def granger_causality(pair1, pair2, close1, close2, max_lag=5):
        """
        Performs a one-way Granger Causality Test from predictor → response using specific interval returns.

        Tests whether past returns of the predictor (X) help forecast the returns of the response (Y).
        Returns detailed stats including significant lags, best lag, and the dominant-follower relation.

        Parameters:
        -----------
        predictor_candles : Candles
        response_candles : Candles
        max_lag : int, default=6
            Maximum number of lags (in 5-minute steps) to test.

        Returns:
        --------
        result : dict
            Dictionary containing:
            - 'predictor': trading pair name of X
            - 'response': trading pair name of Y
            - 'p_values': dictionary of {lag: p-value}
            - 'significant_lags': dictionary of {lag: p-value} where p < 0.05
            - 'causal': True if any lag is significant
            - 'best_lag': lag with lowest p-value (if significant)
            - 'best_p_value': the corresponding lowest p-value (if significant)
            - 'dominant': predictor trading pair if significant, else None
            - 'follower': response trading pair if significant, else None

        Notes:
        ------
        - Uses returns via `.pct_change()` and aligns series.
        - A p-value < 0.05 at any lag means the predictor Granger-causes the response.
        """
        # Align lengths
        min_len = min(len(close1), len(close2))
        x = close1[-min_len:]
        y = close2[-min_len:]

        # Prepare for Granger test
        df = pd.concat([y, x], axis=1)
        df.columns = ['Y', 'X']

        # Run test
        test_result = grangercausalitytests(df, maxlag=max_lag, verbose=False)

        # Extract p-values
        p_values = {
            lag: round(stat[0]['ssr_ftest'][1], 4)
            for lag, stat in test_result.items()
        }

        # Filter significant
        significant = {str(lag): p for lag, p in p_values.items() if p < 0.05}

        # Initialize result
        result = {
            'predictor': pair1,
            'response': pair2,
            'p_values': {str(key): value for key, value in p_values.items()},
            'significant_lags': significant,
            'causal': bool(significant),
            'dominant': None,
            'follower': None,
        }

        if significant:
            best_lag = min(significant, key=significant.get)
            result['best_lag'] = int(best_lag)
            result['best_p_value'] = float(significant[best_lag])
            result['dominant'] = pair1
            result['follower'] = pair2

        return result

    @staticmethod
    def dtw_distance_analysis(pair1, pair2, close1, close2):
        """
        Computes the Dynamic Time Warping (DTW) distance between two trading pairs over 5-minute candles.

        DTW measures similarity in shape between two time series, allowing for flexible time alignment.
        The method assumes the trading pair with lower volatility (smoother path) as dominant.

        Parameters:
        -----------
        candle1 : Candles
        candle2 : Candles

        Returns:
        --------
        result : dict
            Dictionary with:
            - 'pair_1': trading pair from candles1
            - 'pair_2': trading pair from candles2
            - 'dtw_distance': DTW distance between the aligned time series
            - 'dominant': assumed leader (lower-volatility series)
            - 'follower': assumed follower

        Notes:
        ------
        - DTW distance is symmetric (no direction).
        - Dominant is inferred by lower standard deviation in the preprocessed series.
        """
        # Align
        min_len = min(len(close1), len(close2))
        s1 = close1[-min_len:]
        s2 = close2[-min_len:]

        # DTW distance
        distance = float(dtw.distance(s1.values, s2.values))

        # Dominance by volatility
        std1 = s1.std()
        std2 = s2.std()

        if std1 < std2:
            dominant = pair1
            follower = pair2
        elif std2 < std1:
            dominant = pair2
            follower = pair1
        else:
            dominant = follower = None  # equal volatility → unclear

        return {
            'dtw_distance': round(distance, 6),
            'dominant': dominant,
            'follower': follower
        }

    @staticmethod
    def transfer_entropy_analysis(pair1, pair2, close1, close2, k=1, bins=3):
        """
        Computes one-way Transfer Entropy from predictor → response using symbolic 5-minute data.

        Transfer Entropy (TE) quantifies how much information past values of one asset (predictor)
        contribute to predicting the future of another (response), even for non-linear dependencies.

        Parameters:
        -----------
        predictor_candles : Candles
        response_candles : Candles
        k : int, default=1
            History length (order) to use in the TE calculation.
        bins : int, default=3
            Number of bins to use for symbolic discretization.

        Returns:
        --------
        result : dict
            Dictionary containing:
            - 'predictor': name of the source asset (X)
            - 'response': name of the target asset (Y)
            - 'transfer_entropy': float value of TE(X → Y)
            - 'dominant': predictor if TE > 0, else None
            - 'follower': response if TE > 0, else None
            - 'symbolic_bins': number of bins used
            - 'history_k': k used for TE calculation

        Notes:
        ------
        - TE is directional: TE(X→Y) ≠ TE(Y→X)
        - Output values are ≥ 0; higher = more information transfer
        - Returns must be discretized for symbolic entropy-based methods
        """
        # Align lengths
        min_len = min(len(close1), len(close2))
        s1 = close1[-min_len:]
        s2 = close2[-min_len:]

        # Symbolic binning
        s1_binned = np.digitize(s1, bins=np.histogram_bin_edges(s1, bins=bins)) - 1
        s2_binned = np.digitize(s2, bins=np.histogram_bin_edges(s2, bins=bins)) - 1

        x = s1_binned.astype(int).tolist()
        y = s2_binned.astype(int).tolist()

        # Compute TE from X → Y
        try:
            te = round(transfer_entropy(x, y, k), 6)
        except Exception as e:
            te = None

        return {
            'predictor': pair1,
            'response': pair2,
            'transfer_entropy': float(te),
            'dominant': pair1 if te and te > 0 else None,
            'follower': pair2 if te and te > 0 else None,
            'symbolic_bins': bins,
            'history_k': k
        }

    def analyze_pair_cointegration(self, y_col, x_col, interval: str = "15m"):
        """Comprehensive cointegration analysis combining spread analysis and trading signals."""
        interval_multiplier = {
            "1m": 60 * 24,
            "3m": (60 / 3) * 24,
            "5m": (60 / 5) * 24,
            "15m": (60 / 15) * 24,
            "30m": (60 / 15) * 24,
            "1h": 24,
            "4h": 24 / 4
        }
        lookback_days = self.config.get("lookback_days", 14)

        # Calculate periods for 15m candles
        lookback_periods = lookback_days * int(interval_multiplier[interval])

        # Prepare price series
        y_col = y_col.dropna()
        x_col = x_col.dropna()

        # Ensure finite values
        y_col = y_col[np.isfinite(y_col)]
        x_col = x_col[np.isfinite(x_col)]

        # Get last n periods for analysis
        y_col = y_col.tail(lookback_periods)
        x_col = x_col.tail(lookback_periods)

        # Ensure both series are of the same length
        min_len = min(len(y_col), len(x_col))
        y_col = y_col[-min_len:]
        x_col = x_col[-min_len:]

        y, x = y_col.values, x_col.values

        # Run Engle-Granger test
        coint_res: Tuple = coint(y, x)
        p_value = coint_res[1]

        return {
            'p_value': p_value,
            'lookback_days_timestamp': y_col.index.min().timestamp(),
        }


async def main():
    days_of_data = 15
    candles_config = dict(connector_name='binance_perpetual',
                          interval='15m',
                          days=days_of_data,
                          batch_size=20,
                          sleep_time=5.0)
    task_config = {
        "connector_name": "binance_perpetual",
        "quote_asset": "USDT",
        "mongo_uri": os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/"),
        "candles_config": candles_config,
        "update_candles": False,
        "volume_quantile": 0.75,
        "lookback_days": days_of_data,
        "max_lookback_steps": 3,
        "lookback_step": 4 * 24 * 5,
        "p_value_threshold": 0.05,
    }
    task = CointegrationV2Task(name="cointegration_task_v2",
                               frequency=timedelta(hours=1),
                               config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
