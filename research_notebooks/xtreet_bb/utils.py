from typing import List

import pandas as pd
import pandas_ta as ta  # noqa: F401
import yaml

from data_structures.candles import Candles


def generate_report(candles: List[Candles], BOLLINGER_LENGTH: float, BOLLINGER_STD: float):
    report = []

    def apply_signal(x):
        if x >= 1:
            return -1
        elif x <= 0:
            return 1
        else:
            return 0

    for candle in candles:
        try:
            # Assuming df has already been created and processed as per your initial steps
            df = candle.data
            df.ta.bbands(length=BOLLINGER_LENGTH, std=BOLLINGER_STD, append=True)
            df["signal"] = df[f"BBP_{BOLLINGER_LENGTH}_{BOLLINGER_STD}"].apply(apply_signal)
            df["out_of_bounds"] = df["signal"].diff().fillna(0)
            df.loc[df["signal"] == 0, "out_of_bounds"] = 0
            df["volume_usd"] = df["volume"] * df["close"]

            # Create df_filtered for values where out_of_bounds is not 0
            df_filtered = df[df["out_of_bounds"] != 0].copy()
            df["score"] = 0

            score = (df.loc[df["out_of_bounds"] != 0, "out_of_bounds"].shift() != df.loc[
                df["out_of_bounds"] != 0, "out_of_bounds"]).cumsum()

            n_out_of_bounds = len(df[df["out_of_bounds"] != 0])

            df.loc[df_filtered.index, "score"] = score
            df_filtered = df[df["score"] != 0].copy()
            df_filtered["changes"] = df_filtered["score"].shift() != df_filtered["score"]
            timestamps_change = df_filtered[df_filtered["changes"] != 0].index

            periods = []

            for i, timestamp in enumerate(timestamps_change):
                if i == len(timestamps_change) - 1:
                    break
                next_timestamp = timestamps_change[i + 1]
                df_temp = df[timestamp:next_timestamp].copy()
                max_price = df_temp["high"].max()
                min_price = df_temp["low"].min()
                start_price = df_temp["close"].iloc[0]
                end_price = df_temp["close"].iloc[-1]
                signal = df_temp["signal"].iloc[0]
                if signal == 1:
                    cross_reversion = (end_price - start_price) / start_price
                    worst_deviation = abs(min_price - start_price) / start_price
                    max_reversion = (max_price - start_price) / start_price
                elif signal == -1:
                    cross_reversion = -1 * (end_price - start_price) / start_price
                    worst_deviation = abs(max_price - start_price) / start_price
                    max_reversion = -1 * (min_price - start_price) / start_price
                else:
                    continue

                periods.append({
                    "start_timestamp": timestamp,
                    "end_timestamp": next_timestamp,
                    "start_price": start_price,
                    "max_price": max_price,
                    "min_price": min_price,
                    "end_price": end_price,
                    "signal": signal,
                    "worst_deviation": worst_deviation,
                    "max_reversion": max_reversion,
                    "cross_reversion": cross_reversion,
                    "risk_ratio": max_reversion / worst_deviation if worst_deviation != 0 else 1
                })
            total_reversion = [price["cross_reversion"] for price in periods]

            fake_reversions = [r for r in total_reversion if r < 0]
            right_reversions = [r for r in total_reversion if r > 0]
            worst_deviations = [price["worst_deviation"] for price in periods]
            max_reversions = [price["max_reversion"] for price in periods]

            # Metrics
            worst_q0 = pd.Series(worst_deviations).quantile(0)
            worst_q1 = pd.Series(worst_deviations).quantile(0.25)
            worst_q2 = pd.Series(worst_deviations).quantile(0.5)
            worst_q3 = pd.Series(worst_deviations).quantile(0.75)
            worst_q4 = pd.Series(worst_deviations).quantile(1)

            max_rev_q0 = pd.Series(max_reversions).quantile(0)
            max_rev_q1 = pd.Series(max_reversions).quantile(0.25)
            max_rev_q2 = pd.Series(max_reversions).quantile(0.5)
            max_rev_q3 = pd.Series(max_reversions).quantile(0.75)
            max_rev_q4 = pd.Series(max_reversions).quantile(1)

            n_right_reversions = len(right_reversions)
            n_fake_reversions = len(fake_reversions)
            street_cross = len(periods)
            risk_ration_mean = pd.Series([price["risk_ratio"] for price in periods]).mean()

            report.append({
                "trading_pair": candle.trading_pair,
                "worst_q0": worst_q0,
                "worst_q1": worst_q1,
                "worst_q2": worst_q2,
                "worst_q3": worst_q3,
                "worst_q4": worst_q4,
                "max_rev_q0": max_rev_q0,
                "max_rev_q1": max_rev_q1,
                "max_rev_q2": max_rev_q2,
                "max_rev_q3": max_rev_q3,
                "max_rev_q4": max_rev_q4,
                "n_right_reversions": n_right_reversions,
                "n_fake_reversions": n_fake_reversions,
                "street_cross": street_cross,
                "n_out_of_bounds": n_out_of_bounds,
                "risk_ration_mean": risk_ration_mean,
                "volume_mean": df["volume_usd"].mean(),
                "volume_median": df["volume_usd"].median(),
                "returns_std": df['close'].pct_change().std(),
                "right_fake_ratio": n_right_reversions / n_fake_reversions
            })
        except Exception as e:
            print(f"Error processing {candle.trading_pair}: {e}")
            continue

    # Convert report to DataFrame
    report_df = pd.DataFrame(report)
    return report_df


def filter_top_markets(report_df, top_x):
    # Filter and sort by criteria
    filtered_df = report_df.copy()
    # filtered_df = report_df[(report_df['average_volume_per_hour'] > min_volume_usd) &
    #                         (report_df['mean_natr'] > min_atr) &
    #                         (report_df['latest_trend'] > trend_threshold)]
    top_markets_df = filtered_df.sort_values(by='risk_ration_mean', ascending=False).head(top_x)
    return top_markets_df


def distribute_total_amount(top_markets: pd.DataFrame, total_amount: float) -> pd.Series:
    num_markets = len(top_markets)
    amount_per_market = total_amount / num_markets
    return amount_per_market


def generate_config(connector_name: str, intervals: List[str], top_markets: pd.DataFrame, total_amount: float,
                    max_executors_per_side: int, cooldown_time: int, leverage: int,
                    time_limit: int,
                    bb_lengths: List[int], bb_stds: List[float], sl_std_multiplier: float) -> List[dict]:
    # Distribute the total amount based on the score
    top_markets['total_amount_quote'] = distribute_total_amount(top_markets, total_amount)

    configs = []
    for index, row in top_markets.iterrows():
        for bb_length in bb_lengths:
            for bb_std in bb_stds:
                for interval in intervals:
                    trading_pair = row['trading_pair']
                    worst_dev_cols = [f"worst_q{i}" for i in range(1, 5)]
                    for worst_dev_col in worst_dev_cols[1:3]:
                        stop_loss = row[worst_dev_col]
                        all_spreads = calculate_dca_spreads(row)
                        all_reversions = calculate_reversions(row)
                        for dca_spreads in all_spreads:
                            s0, s1 = dca_spreads[0], dca_spreads[1]
                            for reversions in all_reversions:
                                r1, r2 = reversions[0], reversions[1]
                                take_profit = r1
                                if stop_loss <= 0 or take_profit <= 0:
                                    continue
                                a_1 = (s1 - s0) / (r2 - r1) - 1
                                if a_1 <= 1:
                                    continue
                                dca_amounts = [1, a_1]
                                bep = ((1 * 1 + a_1 * (s1 + 1)) / (1 + a_1)) - 1
                                if (bep + stop_loss) * 1.05 <= s1:
                                    print(f"Skipping {trading_pair} due to stop loss closer to get executed:"
                                          f"BEP: {bep}, SL: {stop_loss}, S1: {s1}")
                                    continue
                                # Generate the configuration
                                id = f"xtreet_bb_{connector_name}_{interval}_{trading_pair}_" \
                                     f"bb{bb_length}_{bb_std}_sl{round(100 * stop_loss, 1)}_tp{round(100 * take_profit, 1)}" \
                                     f"bep{bep}"
                                config = {"controller_name": "xtreet_bb", "controller_type": "directional_trading",
                                          "manual_kill_switch": None, "candles_config": [],
                                          "trading_pair": trading_pair, "connector_name": connector_name, "interval": interval,
                                          "max_executors_per_side": max_executors_per_side, "cooldown_time": cooldown_time,
                                          "leverage": leverage, "position_mode": "HEDGE", "time_limit": time_limit,
                                          "dca_spreads": dca_spreads, "dca_amounts_pct": dca_amounts, "bb_length": bb_length,
                                          "bb_std": bb_std, "take_profit": take_profit, "stop_loss": stop_loss,
                                          "total_amount_quote": row['total_amount_quote'],
                                          "candles_trading_pair": trading_pair,
                                          "candles_connector": connector_name,
                                          "id": id}

                                configs.append(config)
    return configs


def calculate_dca_spreads(row):
    columns = [f"worst_q{i}" for i in range(1, 5)]
    all_spreads = []
    for column in columns:
        spreads = [-0.01, row[column]]
        all_spreads.append(spreads)
    return all_spreads


def calculate_reversions(row):
    all_reversions = []
    for i in range(1, 4):
        for j in range(1, 4):
            if i < j:
                all_reversions.append([row[f"max_rev_q{i}"], row[f"max_rev_q{j}"]])
    return all_reversions


def dump_dict_to_yaml(folder: str, data_dict: dict):
    with open(folder + data_dict["id"] + ".yml", 'w') as file:
        yaml.dump(data_dict, file, sort_keys=False)


def read_yaml_to_dict(file_path: str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
