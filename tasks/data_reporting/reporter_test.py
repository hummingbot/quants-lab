import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
from email.mime.base import MIMEBase
from email import encoders

# Database credentials
username = 'admin'
password = 'admin'
host = 'localhost'
port = '5432'  # Default PostgreSQL port
database = 'timescaledb'

# Connection string
connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'
engine = create_engine(connection_string)
connection = engine.connect()


def get_base():
    query = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
    """
    return pd.read_sql(query, engine)


def query_str(cond='>=', c="binance_perpetual_xvg_usdt_trades", day='CURRENT_DATE'):
    return f"""
        SELECT count(*) trade_amount, 
            avg(price) price_avg, 
            max(price) price_max, 
            min(price) price_min, 
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS price_median, 
            max(timestamp) max_timestamp
        FROM {c}
        WHERE timestamp::DATE {cond} {day}
    """


def get_df_query(day="CURRENT_DATE", c="binance_perpetual_xvg_usdt_trades"):
    data = pd.read_sql(query_str(day=day, c=c), engine)
    print("\nData Queried Successfully\n")
    print(data)
    return data


def get_df_query_previous_data(day="CURRENT_DATE", c="binance_perpetual_xvg_usdt_trades"):
    data = pd.read_sql(query_str(cond='<', day=day, c=c), engine)
    # print(data)
    return data


def tune_df(df, c, when='today'):
    df['trading_pair'] = c
    df['when'] = when
    return pd.DataFrame(df, df.index)


def get_final_df(trading_pairs):
    final_df = pd.DataFrame()
    for i, c in enumerate(trading_pairs):
        print(
            f"\nTrading Pair: {c} \nProgress: {i + 1}/{len(trading_pairs)} ({str(i / len(trading_pairs) * 100)[:5]}%)")
        df = get_df_query(c=c)
        df['is_new'] = False
        if df['trade_amount'].sum() > 0:
            if get_df_query_previous_data(c=c)['trade_amount'].sum() == 0:
                df['is_new'] = True
            final_df = pd.concat([final_df, tune_df(df, c)])
        else:
            df = get_df_query(day="CURRENT_DATE - INTERVAL '1 day'", c=c)
            df['is_new'] = False
            if df['trade_amount'].sum() > 0:
                final_df = pd.concat([final_df, tune_df(df, c, when='yesterday')])

    final_df['abs_diff'] = np.maximum(
        (final_df['price_max'] - final_df['price_median']) / final_df['price_median'],
        (final_df['price_median'] - final_df['price_min']) / final_df['price_median']
    )
    final_df.to_csv("trades_analizer_output.csv", index=False)
    return final_df


def print_str(trading_pairs, final_df):
    missing_pairs_list = [c for c in trading_pairs if c not in final_df['trading_pair'].unique() and c != "trades"]
    outdated_pairs_list = [c for c in trading_pairs if
                           c in final_df[final_df['when'] == 'yesterday']['trading_pair'].unique() and c != "trades"]
    correct_trading_pairs_list = [c for c in trading_pairs if
                                  c in final_df[final_df['when'] == 'today']['trading_pair'].unique() and c != "trades"]
    new_trading_pairs_list = [c for c in trading_pairs if
                              c in final_df[final_df['is_new'] == True]['trading_pair'].unique() and c != "trades"]

    missing_trading_pairs = (
        "\n".join(f' ----- {c}' for c in missing_pairs_list)
        if len(missing_pairs_list) < 25
        else "----- Too many trading pairs missing, check missing_trading_pairs.csv for more info"
    )
    outdated_trading_pairs = (
        "\n".join(f' ----- {c}' for c in outdated_pairs_list)
        if len(outdated_pairs_list) < 25
        else "----- Too many outdated trading pairs, check outdated_trading_pairs.csv for more info"
    )

    correct_trading_pairs = (
        "\n".join(f' ----- {c}' for c in correct_trading_pairs_list)
        if len(correct_trading_pairs_list) < 25
        else "----- Too many correct trading pairs, check correct_trading_pairs.csv for more info"
    )

    new_trading_pairs = (
        "\n".join(f' ----- {c}' for c in new_trading_pairs_list)
        if len(new_trading_pairs_list) < 25
        else "----- Too many new trading pairs, check new_trading_pairs.csv for more info"
    )

    report = f"\n\nHello Mr. Pickantell,\n\nHere's a quick review on the database '{database}', stored in {host}:"
    if missing_trading_pairs + outdated_trading_pairs == "":
        report += "\n\n --> Great job! Every trading pair in the database has been updated today."
    if missing_trading_pairs != "":
        report += f"\n\n --> Some trading pairs have not been updated for 2 days:\n{missing_trading_pairs}"
    if outdated_trading_pairs != "":
        report += f"\n\n --> Trading pairs that have not been updated since yesterday (outdated):\n{outdated_trading_pairs}"
    if correct_trading_pairs_list != "":
        report += f"\n\n --> This trading pairs have been correctly updated today:\n{correct_trading_pairs_list}"
    if new_trading_pairs != "":
        report += f"\n\n --> These are the new trading pairs, uploaded today for the first time:\n{new_trading_pairs}"
    else:
        report += "\n\n --> No new trading pairs registered.."

    top_markets = final_df[final_df['abs_diff'] > final_df['abs_diff'].quantile(0.95)].sort_values('abs_diff',
                                                                                                   ascending=False)
    top_markets['trading_pair'] = top_markets['trading_pair'].str.replace("_trades", "", regex=True)
    top_markets[
        ['trading_pair', 'abs_diff', 'price_min', 'price_median', 'price_avg', 'price_max', 'trade_amount', 'is_new',
         'max_timestamp']].to_csv("top_price_deviation_mkts.csv")

    missing_perc = len(missing_pairs_list) / len(trading_pairs)
    outdated_perc = len(outdated_pairs_list) / len(trading_pairs)
    correct_perc = len(correct_trading_pairs_list) / len(trading_pairs)
    new_perc = len(new_trading_pairs_list) / len(trading_pairs)

    report += "\n\n\nAdditional Database Flux Information:\n"
    report += f"\n--> Amount of trading pairs missing (no info for 2 days) out of total pairs: {missing_perc * 100:,.2f}%\n"
    report += f"\n--> Outdated pairs (no info since yesterday) out of total pairs: {outdated_perc * 100:,.2f}%\n"
    report += f"\n--> Correct pairs (updated info) out of total pairs: {correct_perc * 100:,.2f}%\n"
    report += f"\n--> New pairs out of total pairs: {new_perc * 100:,.2f}%\n"
    report += "\n\nFor more information visit the attached files: \n ++ trades_analizer_output.csv: general information about current databases \n ++ top_price_deviation_mkts.csv: Hot markets with nice tolles"
    report += "\n\nSee you soon and don't forget to be awesome!!"

    dict_report = {
        "missing_pairs_df": final_df[final_df['trading_pair'].isin(missing_pairs_list)],
        "outdated_pairs_df": final_df[final_df['trading_pair'].isin(outdated_pairs_list)],
        "correct_trading_pairs_df": final_df[final_df['trading_pair'].isin(correct_trading_pairs_list)],
        "new_trading_pairs_df": final_df[final_df['trading_pair'].isin(new_trading_pairs_list)],
    }

    return report, {k: v for k, v in dict_report.items() if len(v) > 25}


def get_part_attach(path=""):
    with open(path, "rb") as attachment:
        # Set up the MIMEBase with the file
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    # Encode the payload in Base64
    encoders.encode_base64(part)

    # Add header to the attachment
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {os.path.basename(path)}",
    )
    return part


# Get the base and trading pairs
base = get_base()
trading_pairs = base[base['table_name'].str.contains('_trades')]['table_name'].unique()
final_df = get_final_df(trading_pairs)

try:
    # Email configuration
    sender_email = 'thinkingscience.ts@gmail.com'
    recipients = ['palmiscianoblas@gmail.com']
    # recipients = ['palmiscianoblas@gmail.com','federico.cardoso.e@gmail.com', 'tomasgaudino8@gmail.com', 'apelsantiago@gmail.com']
    subject = "Database Refresh Report - Thinking Science Journal"

    # Set up the MIME
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = ", ".join(recipients)
    message['Subject'] = subject

    # Attach the generated report
    generated_report, csv_dict = print_str(trading_pairs, final_df)
    message.attach(MIMEText(generated_report, 'plain'))  # Use 'plain' or 'html' depending on formatting

    # Attach the CSV file to the message
    message.attach(get_part_attach("trades_analizer_output.csv"))
    message.attach(get_part_attach("top_price_deviation_mkts.csv"))

    for c in csv_dict.keys():
        csv_dict[c].to_csv(f"{c}.csv")
        message.attach(get_part_attach(f"{c}.csv"))

    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, 'dqtn zjkf aumv esak')  # App password used here
        server.sendmail(sender_email, recipients, message.as_string())

    print('Email sent successfully')

except Exception as e:
    print(f"Error: {e}")

# Clean up
connection.close()
engine.dispose()

