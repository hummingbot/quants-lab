from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import logging
import asyncio
import os
import sys
from typing import List, Dict, Any, Tuple

from scipy import stats
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from core.data_sources import CLOBDataSource
from core.data_structures.candles import Candles
from core.services.mongodb_client import MongoClient
from core.task_base import BaseTask

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class CointegrationTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.mongo_client = MongoClient(**config.get("db_config", {}))
        self.root_path = "../../.."
        self.clob = CLOBDataSource()

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()

    async def cleanup(self):
        """Cleanup resources."""
        await self.mongo_client.disconnect()

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            candles = await self.get_candles()
            cointegration_results: List[Dict[str, Any]] = self.analyze_trading_pairs(candles)

            await self.mongo_client.insert_documents(collection_name="cointegration_results",
                                                     documents=cointegration_results,
                                                     index=[("base", 1), ("quote", 1)])

            logging.info(f"Successfully added {len(cointegration_results)} cointegration records")

        except Exception as e:
            logging.error(f"Error in Cointegration Task: {str(e)}")
            raise

    async def get_candles(self):
        trading_rules = await self.clob.get_trading_rules(connector_name=self.config["connector_names"][0])
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

    def get_filtered_candles_by_volume_quantile(self, candles: List[Candles]):
        all_candles = pd.DataFrame()
        for candle in candles:
            df = candle.data.copy()
            df["trading_pair"] = candle.trading_pair
            all_candles = pd.concat([all_candles, df])
        grouped_candles = all_candles.groupby("trading_pair")["quote_asset_volume"].sum().reset_index()
        volume_filter_quantile = grouped_candles["quote_asset_volume"].quantile(self.config.get("volume_quantile", 0.75))
        selected_candles = grouped_candles[grouped_candles["quote_asset_volume"] >= volume_filter_quantile]
        trading_pairs = [
            candle.trading_pair for candle in candles
            if candle.trading_pair in selected_candles.trading_pair.values
        ]
        return trading_pairs

    def analyze_trading_pairs(self, candles: List[Candles]):
        # Get the analysis DataFrame
        results_df = self.analyze_multiple_pairs(candles)

        # Filter for cointegrated pairs
        pair_results = []
        processed_pairs = set()

        for _, row in results_df.iterrows():
            pair_key = tuple(sorted([row['base'], row['quote']]))

            # Skip if we've already processed this pair
            if pair_key in processed_pairs:
                continue

            # Get both directions for this pair
            pair_analysis = results_df[
                ((results_df['base'] == row['base']) & (results_df['quote'] == row['quote'])) |
                ((results_df['base'] == row['quote']) & (results_df['quote'] == row['base']))
                ]

            # Check if both directions are cointegrated
            if not (pair_analysis['p_value'] < self.config.get("p_value_threshold", 0.05)).all():
                continue

            # Find long and short positions
            long_position = pair_analysis[pair_analysis['side'] == 'long'].iloc[0] if len(
                pair_analysis[pair_analysis['side'] == 'long']) > 0 else None
            short_position = pair_analysis[pair_analysis['side'] == 'short'].iloc[0] if len(
                pair_analysis[pair_analysis['side'] == 'short']) > 0 else None

            # Skip if we don't have both positions or if any grid prices are zero/None
            if (long_position is None or short_position is None or
                    not long_position['entry_price'] or not long_position['end_price'] or not long_position[
                        'limit_price'] or
                    not short_position['entry_price'] or not short_position['end_price'] or not short_position[
                        'limit_price'] or
                    long_position['entry_price'] == 0 or long_position['end_price'] == 0 or long_position[
                        'limit_price'] == 0 or
                    short_position['entry_price'] == 0 or short_position['end_price'] == 0 or short_position[
                        'limit_price'] == 0):
                continue

            # Calculate average cointegration value
            coint_value = pair_analysis['p_value'].mean()

            # Create result dictionary for this pair
            result = {
                'base': long_position['base'],
                'quote': short_position['base'],
                'grid_base': {
                    'start_price': float(long_position['entry_price']),
                    'end_price': float(long_position['end_price']),
                    'limit_price': float(long_position['limit_price']),
                    'beta': float(long_position['beta'])  # Replace max_open_orders with beta
                },
                'grid_quote': {
                    'start_price': float(short_position['entry_price']),
                    'end_price': float(short_position['end_price']),
                    'limit_price': float(short_position['limit_price']),
                    'beta': float(short_position['beta'])  # Replace max_open_orders with beta
                },
                'coint_value': float(coint_value)
            }
            pair_results.append(result)

            # Mark this pair as processed
            processed_pairs.add(pair_key)

        # Sort results by cointegration value
        pair_results.sort(key=lambda x: x['coint_value'])

        # Print summary
        print(f"\nFound {len(pair_results)} cointegrated pairs:")
        print("-" * 50)
        for result in pair_results:
            print(f"\nLong {result['base']} vs Short {result['quote']}")
            print(f"Cointegration value: {result['coint_value']:.4f}")
            print(f"Grid Base - Entry: {result['grid_base']['start_price']:.2f}, "
                  f"End: {result['grid_base']['end_price']:.2f}, "
                  f"Limit: {result['grid_base']['limit_price']:.2f}, "
                  f"Beta: {result['grid_base']['beta']:.4f}")
            print(f"Grid Quote - Entry: {result['grid_quote']['start_price']:.2f}, "
                  f"End: {result['grid_quote']['end_price']:.2f}, "
                  f"Limit: {result['grid_quote']['limit_price']:.2f}, "
                  f"Beta: {result['grid_quote']['beta']:.4f}")

        return pair_results

    def analyze_multiple_pairs(self, candles: List[Candles]):
        # Create lists to store results
        results = []
        trading_pairs = self.get_filtered_candles_by_volume_quantile(candles)
        pair_names = [candle.trading_pair for candle in candles if candle.trading_pair in trading_pairs]
        # Total number of combinations
        total_combinations = (len(pair_names) * (len(pair_names) - 1)) // 2

        # Progress bar for the combinations
        with tqdm(total=total_combinations, desc="Analyzing Cointegration", unit="pair") as pbar:
            # Analyze each possible pair combination
            for i in range(len(pair_names)):
                for j in range(i + 1, len(pair_names)):
                    pair1 = pair_names[i]
                    pair2 = pair_names[j]

                    try:
                        # Get candle data
                        candle1 = next((candle for candle in candles if candle.trading_pair == pair1), None)
                        candle2 = next((candle for candle in candles if candle.trading_pair == pair2), None)

                        # Get normalized price series
                        price1 = candle1.data["close"].pct_change().add(1).cumprod()
                        price2 = candle2.data["close"].pct_change().add(1).cumprod()

                        # Analyze both directions
                        analysis_1vs2 = self.analyze_pair_cointegration(price1, price2, candle1.interval)
                        analysis_2vs1 = self.analyze_pair_cointegration(price2, price1, candle2.interval)

                        # Generate grid levels
                        current_price1 = candle1.data["close"].iloc[-1]
                        current_price2 = candle2.data["close"].iloc[-1]

                        grid_1vs2 = self.generate_grid_levels(analysis=analysis_1vs2, current_price=current_price1)
                        grid_2vs1 = self.generate_grid_levels(analysis=analysis_2vs1, current_price=current_price2)

                        # Store results for first direction
                        results.append({
                            'base': pair1,
                            'quote': pair2,
                            'p_value': analysis_1vs2['p_value'],
                            'z_score': analysis_1vs2['current_z_score'],
                            'side': analysis_1vs2['side'],
                            'signal_strength': analysis_1vs2['signal_strength'],
                            'mean_reversion_prob': analysis_1vs2['mean_reversion_probability'],
                            'beta': analysis_1vs2['beta'],
                            'entry_price': grid_1vs2['entry_price'] if grid_1vs2['side'] != 'both' else None,
                            'end_price': grid_1vs2['end_price'] if grid_1vs2['side'] != 'both' else None,
                            'limit_price': grid_1vs2['limit_price'] if grid_1vs2['side'] != 'both' else None
                        })

                        # Store results for second direction
                        results.append({
                            'base': pair2,
                            'quote': pair1,
                            'p_value': analysis_2vs1['p_value'],
                            'z_score': analysis_2vs1['current_z_score'],
                            'side': analysis_2vs1['side'],
                            'signal_strength': analysis_2vs1['signal_strength'],
                            'mean_reversion_prob': analysis_2vs1['mean_reversion_probability'],
                            'beta': analysis_2vs1['beta'],
                            'entry_price': grid_2vs1['entry_price'] if grid_2vs1['side'] != 'both' else None,
                            'end_price': grid_2vs1['end_price'] if grid_2vs1['side'] != 'both' else None,
                            'limit_price': grid_2vs1['limit_price'] if grid_2vs1['side'] != 'both' else None
                        })
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error analyzing {pair1} vs {pair2}: {str(e)}")
                        continue

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add derived columns
        df['Cointegrated'] = df['p_value'] < self.config.get("p_value_threshold", 0.05)
        df['potential_profit'] = np.where(df['side'] != 'both',
                                          abs(df['end_price'] - df['entry_price']) / df['entry_price'],
                                          0)
        df['Risk_Ratio'] = np.where(df['side'] != 'both',
                                    abs(df['end_price'] - df['entry_price']) /
                                    abs(df['limit_price'] - df['entry_price']),
                                    0)

        # Sort by signal strength and potential profit
        df = df.sort_values(['signal_strength', 'potential_profit'],
                            ascending=[False, False])

        return df

    def generate_grid_levels(self, analysis, current_price):
        """
        Generate grid trading levels based on Z-scores with configurable thresholds.

        Args:
            analysis (dict): Results from cointegration analysis
            current_price (float): Current price of the asset

        Returns:
            dict: Grid trading parameters and levels
        """
        entry_threshold = self.config.get("entry_threshold", 1.5)
        stop_threshold = self.config.get("stop_threshold", 1.0)
        grid_levels = self.config.get("grid_levels", 5)
        time_limit_hours = self.config.get("time_limit_hours", 24)

        z_score = analysis['current_z_score']
        z_std = analysis['z_std']
        beta = analysis['position_ratio']

        # Time parameters
        current_time = analysis['actual_values'].index[-1]
        time_limit = current_time + timedelta(hours=time_limit_hours)

        if abs(z_score) > entry_threshold:
            is_short = z_score > 0

            entry_price = current_price  # TODO: shift or give tolerance to this price

            # Calculate target and stop prices
            if is_short:
                end_price = current_price * (1 - (z_score * z_std * beta))
                limit_price = current_price * (1 + (stop_threshold * z_std * beta))
                grid_direction = -1
            else:  # long
                end_price = current_price * (1 + (abs(z_score) * z_std * beta))
                limit_price = current_price * (1 - (stop_threshold * z_std * beta))
                grid_direction = 1

            # Generate grid levels
            price_range = abs(end_price - entry_price)
            grid_step = price_range / (grid_levels + 1)
            grid_prices = [entry_price + (i * grid_step * grid_direction) for i in range(1, grid_levels + 1)]

            grid = {
                'side': 'short' if is_short else 'long',
                'entry_price': entry_price,
                'end_price': end_price,
                'limit_price': limit_price,
                'grid_prices': grid_prices,
                'time_limit': time_limit,
                'entry_z_score': z_score,
                'target_z_score': 0,
                'stop_z_score': z_score + (stop_threshold * (1 if is_short else -1)),
                'grid_levels': grid_levels,
                'grid_step': grid_step
            }

        else:  # No signal
            grid = {
                'side': 'both',
                'entry_price': None,
                'end_price': None,
                'limit_price': None,
                'grid_prices': [],
                'time_limit': None,
                'entry_z_score': z_score,
                'target_z_score': None,
                'stop_z_score': None,
                'grid_levels': 0,
                'grid_step': None
            }
        return grid

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
        signal_days = self.config.get("signal_days", 3)
        z_score_threshold = self.config.get("z_score_threshold", 2.0)
        
        # Calculate periods for 15m candles
        lookback_periods = lookback_days * int(interval_multiplier[interval])
        signal_periods = signal_days * int(interval_multiplier[interval])

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

        # Perform linear regression
        x_reshaped = x.reshape(-1, 1)
        reg: LinearRegression = LinearRegression().fit(x_reshaped, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]

        # Calculate spread (z_t)
        z_t = pd.Series(y - (alpha + beta * x))
        z_t.index = y_col.index.copy()
        z_mean = np.mean(z_t)
        z_std = np.std(z_t)

        # Get recent data for signal analysis
        y_recent = y_col.tail(signal_periods)
        x_recent = x_col.tail(signal_periods)

        # Calculate recent predictions and spread
        y_pred = alpha + beta * x_recent
        recent_spread = y_recent - y_pred

        # Calculate current Z-score
        current_z_score = (z_t.iloc[-1] - z_mean) / z_std

        # Determine trading side
        if abs(current_z_score) < z_score_threshold:
            side = "both"
        elif current_z_score > z_score_threshold:
            side = "short"  # Y is overvalued
        else:  # current_z_score < -z_score_threshold
            side = "long"  # Y is undervalued

        # Calculate additional metrics
        signal_strength = abs(current_z_score) / z_score_threshold
        mean_reversion_prob = 1 - stats.norm.cdf(abs(current_z_score))

        # Calculate percentage error for recent period
        percentage_error = ((y_pred - y_recent) / y_recent.abs()) * 100
        median_error = np.median(percentage_error)

        return {
            # Cointegration statistics
            'p_value': p_value,
            'alpha': alpha,
            'beta': beta,

            # Spread analysis
            'z_t': z_t,
            'z_mean': z_mean,
            'z_std': z_std,
            'current_z_score': current_z_score,

            # Trading signals
            'side': side,
            'signal_strength': signal_strength,
            'mean_reversion_probability': mean_reversion_prob,

            # Recent performance
            'recent_spread': recent_spread,
            'predictions': y_pred,
            'actual_values': y_recent,
            'median_error': median_error,

            # Risk management
            'stop_loss_z_score': current_z_score * 1.5,  # 50% additional deviation
            'target_z_score': 0,  # Mean reversion target

            # Trade setup
            'position_ratio': beta,
            'z_score_threshold': z_score_threshold
        }


async def main():
    days_of_data = 14
    mongodb_config = {
        "username": os.getenv('MONGO_INITDB_ROOT_USERNAME', "admin"),
        "password": os.getenv('MONGO_INITDB_ROOT_PASSWORD', "admin"),
        "host": os.getenv('MONGO_HOST', 'localhost'),
        "port": os.getenv('MONGO_PORT', 27017),
        "database": "mongodb"
    }
    candles_config = dict(connector_name='binance_perpetual',
                          interval='15m',
                          days=days_of_data,
                          batch_size=20,
                          sleep_time=5.0)
    task_config = {
        "connector_names": ["binance_perpetual"],
        "quote_asset": "USDT",
        "db_config": mongodb_config,
        "candles_config": candles_config,
        "update_candles": True,
        "volume_quantile": 0.75,
        "z_score_threshold": 0.5,
        "lookback_days": days_of_data,
        "signal_days": 3,

        "p_value_threshold": 0.05,
        "entry_threshold": 1.5,
        "stop_threshold": 1.0,
        "grid_levels": 5,
        "time_limit_hours": 24
    }
    task = CointegrationTask(name="cointegration_task",
                             frequency=timedelta(hours=1),
                             config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
