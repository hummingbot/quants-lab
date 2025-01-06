from typing import List

import pandas as pd

from core.features.candles.volatility import Volatility
from core.features.candles.volume import Volume


def generate_report(candles, volatility_config, volume_config):
    report = []
    for candle in candles:
        try:
            candle.add_features([Volatility(volatility_config), Volume(volume_config)])
            df = candle.data

            # Summary statistics for volatility
            mean_natr = df['natr'].mean()

            # Average volume per hour
            total_volume_usd = df['volume_usd'].sum()
            total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
            average_volume_per_hour = total_volume_usd / total_hours

            # Calculate score
            score = mean_natr * average_volume_per_hour
            report.append({
                'trading_pair': candle.trading_pair,
                'mean_natr': mean_natr,
                'sniper_upper_price': df['close'].max(),
                'sniper_lower_price': df['close'].min(),
                'grid_lower_price': df["close"].quantile(0.05),
                'grid_upper_price': df["close"].quantile(0.95),
                'grid_mid_price': df['close'].median(),
                'average_volume_per_hour': average_volume_per_hour,
                'score': score
            })
        except Exception as e:
            print(f"Error processing {candle.trading_pair}: {e}")
            continue

    # Convert report to DataFrame
    report_df = pd.DataFrame(report)

    # Normalize the score
    max_score = report_df['score'].max()
    report_df['normalized_score'] = report_df['score'] / max_score
    report_df.drop(columns=['score'], inplace=True)

    return report_df


def filter_top_markets(report_df, volume_threshold=0.4, volatility_threshold=0.4):
    # Filter and sort by criteria
    natr_percentile = report_df['mean_natr'].quantile(volatility_threshold)
    volume_percentile = report_df['average_volume_per_hour'].quantile(volume_threshold)

    filtered_df = report_df[
        (report_df['mean_natr'] > natr_percentile) &
        (report_df['average_volume_per_hour'] > volume_percentile)
        ]
    top_markets_df = filtered_df.sort_values(by='normalized_score', ascending=False)
    return top_markets_df


def distribute_total_amount(top_markets: pd.DataFrame, total_amount: float, min_amount_per_market: float) -> pd.Series:
    num_markets = len(top_markets)
    min_total_amount = num_markets * min_amount_per_market

    if min_total_amount > total_amount:
        raise ValueError("Total amount is too low to allocate the minimum amount to each market.")

    remaining_amount = total_amount - min_total_amount
    total_score = top_markets['normalized_score'].sum()
    allocated_amounts = (top_markets['normalized_score'] / total_score) * remaining_amount

    # Ensure each market gets at least the minimum amount
    return allocated_amounts + min_amount_per_market


def generate_configs(version: str, connector_name: str, top_markets: pd.DataFrame, total_amount_quote: float,
                     activation_bounds: float, n_levels: int, grid_allocation: float, inventory_buffer: float) -> List:
    # Generate the configuration
    configs = []
    for index, row in top_markets.iterrows():
        trading_pair = row['trading_pair']

        # Generate market configuration
        market_config = {
            "id": f"gs_{trading_pair}_{version}",
            "controller_name": "grid_strike",
            "controller_type": "generic",
            "manual_kill_switch": None,
            "candles_config": [],
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "total_amount_quote": total_amount_quote,
            "grid_upper_price": row['grid_upper_price'],
            "grid_lower_price": row['grid_lower_price'],
            "grid_mid_price": row['grid_mid_price'],
            "n_levels": n_levels,
            "sniper_upper_price": row['sniper_upper_price'],
            "sniper_lower_price": row['sniper_lower_price'],
            "grid_allocation": grid_allocation,
            "inventory_buffer": inventory_buffer,
            "activation_bounds": activation_bounds,
        }

        configs.append(market_config)
    return configs
