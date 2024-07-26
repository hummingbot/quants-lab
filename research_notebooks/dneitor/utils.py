from typing import Any, Dict, List

import pandas as pd
import yaml

from features.candles.trend import Trend
from features.candles.volatility import Volatility
from features.candles.volume import Volume
from features.utils.peak_analyzer import PeakAnalyzer


def generate_report(candles, volatility_config, trend_config, volume_config):
    report = []
    for trading_pair, candle in candles.items():
        try:
            candle.add_features([Volatility(volatility_config), Trend(trend_config), Volume(volume_config)])
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

            # Calculate score
            score = (mean_natr * 0.8 + mean_bb_width * 0.2) * average_volume_per_hour
            report.append({
                'trading_pair': trading_pair,
                'mean_volatility': mean_volatility,
                'mean_natr': mean_natr,
                'mean_bb_width': mean_bb_width,
                'latest_trend': latest_trend,
                'average_volume_per_hour': average_volume_per_hour,
                'score': score
            })
        except Exception as e:
            print(f"Error processing {trading_pair}: {e}")
            continue

    # Convert report to DataFrame
    report_df = pd.DataFrame(report)

    # Normalize the score
    max_score = report_df['score'].max()
    report_df['normalized_score'] = report_df['score'] / max_score
    report_df.drop(columns=['score'], inplace=True)

    return report_df


def filter_top_markets(report_df, top_x, min_volume_usd=0, min_atr=0, trend_threshold=0.5):
    # Filter and sort by criteria
    filtered_df = report_df[(report_df['average_volume_per_hour'] > min_volume_usd) &
                            (report_df['mean_natr'] > min_atr) &
                            (report_df['latest_trend'] > trend_threshold)]
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
                    max_open_orders: int, min_amount_per_market: float) -> Dict[str, Any]:

    # Distribute the total amount based on the score
    top_markets['total_amount_quote'] = distribute_total_amount(top_markets, total_amount, min_amount_per_market)

    # Generate the configuration
    config = {
        "id": id,
        "controller_name": "dneitor",
        "controller_type": "generic",
        "manual_kill_switch": None,
        "candles_config": [],
        "markets": []
    }

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

        # Generate market configuration
        market_config = {
            "connector_name": connector_name,
            "trading_pair": trading_pair,
            "total_amount_quote": round(row['total_amount_quote'], 2),
            "prices": prices,
            "amounts_quote_pct": [amount for amount in amounts_quote_pct],
            "take_profit": take_profit,
            "activation_bounds": activation_bounds,
            "max_open_orders": max_open_orders
        }

        config["markets"].append(market_config)

    return config


def dump_dict_to_yaml(folder: str, data_dict: dict):
    with open(folder + data_dict["id"] + ".yml", 'w') as file:
        yaml.dump(data_dict, file, sort_keys=False)


def read_yaml_to_dict(file_path: str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
