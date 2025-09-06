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
