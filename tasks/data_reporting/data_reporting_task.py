import asyncio
import logging
import os
import smtplib
from datetime import timedelta
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict

import asyncpg
import numpy as np
import pandas as pd

from core.task_base import BaseTask

logging.basicConfig(level=logging.INFO)


class ReportGeneratorTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )

    async def execute_query(self, query: str):
        async with self.pool.acquire() as connection:
            return await connection.fetch(query)

    async def get_base_tables(self):
        query = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
        """
        return await self.execute_query(query)

    async def execute(self):
        await self.connect()
        base = await self.get_base_tables()
        trading_pairs = [table["table_name"] for table in base if table["table_name"].endswith("_trades")]
        final_df = await self.get_final_df(trading_pairs)

        try:
            # Email configuration
            sender_email = 'thinkingscience.ts@gmail.com'
            recipients = ['palmiscianoblas@gmail.com', 'federico.cardoso.e@gmail.com',
                          'tomasgaudino8@gmail.com', 'apelsantiago@gmail.com']
            subject = "Database Refresh Report - Thinking Science Journal"

            # Set up the MIME
            message = MIMEMultipart()
            message['From'] = sender_email
            message['To'] = ", ".join(recipients)
            message['Subject'] = subject

            # Attach the generated report
            generated_report, csv_dict = self.print_str(trading_pairs, final_df)
            message.attach(MIMEText(generated_report, 'plain'))  # Use 'plain' or 'html' depending on formatting

            # Attach the CSV file to the message
            message.attach(self.get_part_attach("trades_analizer_output.csv"))
            message.attach(self.get_part_attach("top_price_deviation_mkts.csv"))

            for c in csv_dict.keys():
                csv_dict[c].to_csv(f"{c}.csv")
                message.attach(self.get_part_attach(f"{c}.csv"))

            smtp_server = 'smtp.gmail.com'
            smtp_port = 587

            # Send the email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(sender_email, 'dqtn zjkf aumv esak')  # App password used here
                server.sendmail(sender_email, recipients, message.as_string())

            print('Email sent successfully')

        except Exception as e:
            logging.error("An error occurred: while running Report Generator%s", e)
            print(f"Error: {e}")

    async def get_final_df(self, trading_pairs):
        final_data = []
        for i, trading_pair in enumerate(trading_pairs):
            today_general_metrics = await self.get_general_metrics(trading_pair=trading_pair,
                                                                   day="CURRENT_DATE", cond=">=")
            previous_general_metrics = await self.get_general_metrics(trading_pair=trading_pair,
                                                                      day="CURRENT_DATE - INTERVAL '1 day'", cond=">=")
            if today_general_metrics['trade_amount'] > 0:
                general_metrics = today_general_metrics
                general_metrics["when"] = "today"
                general_metrics['is_new'] = True if previous_general_metrics['trade_amount'] == 0 else False

            elif previous_general_metrics['trade_amount'] > 0:
                general_metrics = previous_general_metrics
                general_metrics["when"] = "yesterday"
                general_metrics["is_new"] = False
            else:
                print("No data available for this trading pair")
                return
            general_metrics["trading_pair"] = trading_pair
            final_data.append(general_metrics)
        final_df = pd.DataFrame(final_data)
        final_df["price_avg"] = pd.to_numeric(final_df["price_avg"])
        final_df["price_max"] = pd.to_numeric(final_df["price_max"])
        final_df["price_min"] = pd.to_numeric(final_df["price_min"])
        final_df["price_median"] = pd.to_numeric(final_df["price_median"])
        final_df['abs_diff'] = np.maximum(
            (final_df['price_max'] - final_df['price_median']) / final_df['price_median'],
            (final_df['price_median'] - final_df['price_min']) / final_df['price_median']
        )
        final_df.to_csv("trades_analizer_output.csv", index=False)
        return final_df

    def query_str(self, cond='>=', trading_pair="binance_perpetual_xvg_usdt_trades", day='CURRENT_DATE'):
        return f"""
            SELECT count(*) trade_amount,
                avg(price) price_avg,
                max(price) price_max,
                min(price) price_min,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) AS price_median,
                max(timestamp) max_timestamp
            FROM {trading_pair}
            WHERE timestamp::DATE {cond} {day}
        """

    async def get_general_metrics(self, cond=">=", day="CURRENT_DATE", trading_pair="binance_perpetual_xvg_usdt_trades"):
        df_data = await self.execute_query(self.query_str(cond=cond, day=day, trading_pair=trading_pair))
        return dict(df_data[0])

    def tune_df(self, df, c, when='today'):
        df['trading_pair'] = c
        df['when'] = when
        return pd.DataFrame(df, df.index)

    def print_str(self, trading_pairs, final_df):
        missing_pairs_list = [c for c in trading_pairs if c not in final_df['trading_pair'].unique() and c != "trades"]
        outdated_pairs_list = [c for c in trading_pairs if
                               c in final_df[final_df['when'] == 'yesterday'][
                                   'trading_pair'].unique() and c != "trades"]
        correct_trading_pairs_list = [c for c in trading_pairs if
                                      c in final_df[final_df['when'] == 'today'][
                                          'trading_pair'].unique() and c != "trades"]
        new_trading_pairs_list = [c for c in trading_pairs if
                                  c in final_df[final_df['is_new']]['trading_pair'].unique() and c != "trades"]

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

        new_trading_pairs = (
            "\n".join(f' ----- {c}' for c in new_trading_pairs_list)
            if len(new_trading_pairs_list) < 25
            else "----- Too many new trading pairs, check new_trading_pairs.csv for more info"
        )

        report = f"\n\nHello Mr. Pickantell,\n\nHere's a quick review on the database '{self.database}', stored in {self.host}:"
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
            ['trading_pair', 'abs_diff', 'price_min', 'price_median', 'price_avg', 'price_max', 'trade_amount',
             'is_new',
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
        report += "\n\nFor more information visit the attached files: \n ++ trades_analizer_output.csv: " \
                  "general information about current databases \n ++ top_price_deviation_mkts.csv: " \
                  "Hot markets with nice tolles"
        report += "\n\nSee you soon and don't forget to be awesome!!"

        dict_report = {
            "missing_pairs_df": final_df[final_df['trading_pair'].isin(missing_pairs_list)],
            "outdated_pairs_df": final_df[final_df['trading_pair'].isin(outdated_pairs_list)],
            "correct_trading_pairs_df": final_df[final_df['trading_pair'].isin(correct_trading_pairs_list)],
            "new_trading_pairs_df": final_df[final_df['trading_pair'].isin(new_trading_pairs_list)],
        }

        return report, {k: v for k, v in dict_report.items() if len(v) > 25}

    def get_part_attach(self, path=""):
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


async def main():
    task = ReportGeneratorTask(
        name="Report Generator",
        config={
            "host": "localhost",
            "port": 5432,
            "user": "admin",
            "password": "admin",
            "database": "timescaledb",
        },
        frequency=timedelta(hours=12))
    await task.execute()

if __name__ == "__main__":
    asyncio.run(main())
