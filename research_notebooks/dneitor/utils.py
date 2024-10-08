from typing import List

import pandas as pd
import yaml

from core.features.candles.peak_analyzer import PeakAnalyzer
from core.features.candles.volatility import Volatility
from core.features.candles.volume import Volume
from hummingbot.core.data_type.common import OrderType


def generate_report(candles, volatility_config, volume_config):
    report = []
    for candle in candles:
        try:
            candle.add_features([Volatility(volatility_config), Volume(volume_config)])
            df = candle.data

            # Summary statistics for volatility
            mean_volatility = df['volatility'].mean()
            mean_natr = df['natr'].mean()
            mean_bb_width = df['bb_width'].mean()

            # Latest trend score
            latest_trend = df['rolling_buy_sell_imbalance_short'].iloc[-1]

            # Average volume per hour
            total_volume_usd = df['volume_usd'].sum()
            total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
            average_volume_per_hour = total_volume_usd / total_hours
            max_price = df['close'].max()
            min_price = df['close'].min()
            range_price = max_price - min_price
            range_price_pct = (max_price - df['close'].iloc[-1]) / df['close'].iloc[-1]
            current_position = (max_price - df['close'].iloc[-1]) / range_price

            # Calculate score
            score = mean_natr * average_volume_per_hour * current_position * range_price_pct
            report.append({
                'trading_pair': candle.trading_pair,
                'mean_volatility': mean_volatility,
                'mean_natr': mean_natr,
                'mean_bb_width': mean_bb_width,
                'latest_trend': latest_trend,
                'average_volume_per_hour': average_volume_per_hour,
                'current_position': current_position,
                'range_price_pct': range_price_pct,
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


def filter_top_markets(report_df, top_x, min_volume_usd=0, min_natr=0, current_position_threshold=0.5):
    # Filter and sort by criteria
    filtered_df = report_df[(report_df['average_volume_per_hour'] > min_volume_usd) &
                            (report_df['mean_natr'] > min_natr) &
                            (report_df['current_position'] > current_position_threshold)]
    top_markets_df = filtered_df.sort_values(by='normalized_score', ascending=False).head(top_x)
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


def generate_config(id: str, connector_name: str, candles: pd.DataFrame,  top_markets: pd.DataFrame, total_amount: float,
                    amounts_quote_pct: List[float], activation_bounds: float, take_profit_multiplier: float,
                    max_open_orders: int, min_order_amount: float, leverage: int,
                    take_profit_order_type: OrderType = OrderType.LIMIT_MAKER) -> List:

    # Distribute the total amount based on the score
    top_markets['total_amount_quote'] = total_amount
    configs = []

    for index, row in top_markets.iterrows():
        trading_pair = row['trading_pair']
        candles_df = candles[trading_pair].data

        # Initialize PeakAnalyzer
        analyzer = PeakAnalyzer(candles_df)

        # Extract low price clusters
        _, _, _, low_clusters = analyzer.get_peaks_and_clusters(
            prominence_percentage=0.02,
            distance=5,
            num_clusters=len(amounts_quote_pct) - 1)
        max_price = candles_df['close'].max()
        prices = [float(price) for price in [max_price] + low_clusters]

        # Calculate take profit using NATR
        mean_natr = row['mean_natr']
        take_profit = mean_natr * take_profit_multiplier

        # Generate the configuration
        config = {
            "id": f"{id}_{trading_pair}",
            "controller_name": "dneitor",
            "controller_type": "generic",
            "manual_kill_switch": None,
            "candles_config": [],
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "total_amount_quote": round(row['total_amount_quote'], 2),
            "prices": prices,
            "amounts_quote_pct": [amount for amount in amounts_quote_pct],
            "take_profit": take_profit,
            "activation_bounds": activation_bounds,
            "max_open_orders": max_open_orders,
            "min_order_amount": min_order_amount,
            "leverage": leverage,
            "time_limit": None,
            "take_profit_order_type": take_profit_order_type.value,
        }
        configs.append(config)
    return configs


def dump_dict_to_yaml(folder: str, data_dict: dict):
    with open(folder + data_dict["id"] + ".yml", 'w') as file:
        yaml.dump(data_dict, file, sort_keys=False)


def read_yaml_to_dict(file_path: str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
