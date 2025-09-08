import asyncio
import json
import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import plotly.express as px
from dotenv import load_dotenv

import asyncpg
import pandas as pd
import numpy as np

from core.services.backend_api_client import BackendAPIClient
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask

load_dotenv()
logging.basicConfig(level=logging.INFO)


# Base class for common functionalities like database connection and email sending
class TaskBase(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.name = name
        self.frequency = frequency
        self.config = config
        self.export = self.config.get("export", False)
        self.ts_client = TimescaleClient(host=self.config.get("host", "localhost"))
        self.backend_api_client = BackendAPIClient(host=self.config.get("backend_api_host", "localhost"))

    async def execute_query(self, query: str):
        """Executes a query and returns the result."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query)

    def create_email(self, subject: str, sender_email: str, recipients: List[str], body: str) -> MIMEMultipart:
        """Creates a basic email structure."""
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = ", ".join(recipients)
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))
        return message

    def add_attachment(self, message: MIMEMultipart, path: str):
        """Attaches a file to the email."""
        with open(path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(path)}")
        message.attach(part)

    def send_email(self, message: MIMEMultipart, sender_email: str, app_password: str, smtp_server="smtp.gmail.com",
                   smtp_port=587):
        """Sends an email using the specified SMTP server."""
        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, app_password)
                server.sendmail(sender_email, message["To"].split(", "), message.as_string())
            logging.info("Email sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send email: {e}")


class ReportGeneratorTask(TaskBase):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.base_metrics = None

    async def get_base_tables(self):
        available_pairs = await self.ts_client.get_available_pairs()
        table_names = [self.ts_client.get_trades_table_name(connector_name, trading_pair)
                       for connector_name, trading_pair in available_pairs]
        return table_names

    @staticmethod
    def is_new(row):
        last_data_today = pd.to_datetime(row['to_timestamp']).timestamp() >= datetime.now().timestamp()
        last_data_yesterday = pd.to_datetime(row['to_timestamp']).timestamp() >= (datetime.now() - timedelta(days=1)).timestamp()
        first_data_yesterday = pd.to_datetime(row['from_timestamp']).timestamp() >= (datetime.now() - timedelta(days=1)).timestamp()

        if last_data_today:
            when = 'today'
        elif last_data_yesterday:
            when = 'yesterday'
        else:
            when = None

        row['is_new'] = last_data_today & first_data_yesterday
        row['when'] = when

    async def set_base_metrics(self):
        base_metrics = await self.ts_client.get_db_status_df()
        base_metrics["table_names"] = base_metrics.apply(lambda x: self.ts_client.get_trades_table_name(x["connector_name"], x["trading_pair"]), axis=1)
        base_metrics['when'] = None
        base_metrics['is_new'] = False
        base_metrics.apply(self.is_new, axis=1)
        self.base_metrics = base_metrics.dropna(subset=["when"]).copy()

    async def generate_heatmap(self):
        # Load the all_daily_metrics CSV into a DataFrame
        base_metrics = self.base_metrics
        # Calculate total trade amounts by trading pair and percentage
        total_trade_amounts = base_metrics.groupby('trading_pair')['trade_amount'].transform('sum')
        base_metrics['trade_amount_pct'] = (base_metrics['trade_amount'] / total_trade_amounts) * 100

        # Pivot data for heatmap-ready format with trade_amount as percentage
        heatmap_data = base_metrics.pivot(index='trading_pair', columns='day', values='trade_amount_pct')

        # Create the heatmap using Plotly
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale="Reds",
            labels={'color': 'Trade Amount (%)'},
            title="Trade Amount Heatmap by Date and Trading Pair"
        )

        # Customize the layout for readability
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Trading Pair",
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            xaxis_tickfont_size=12,
            yaxis_tickfont_size=12
        )
        pdf_filename = "trade_amount_heatmap.pdf"
        # Save to PDF if required
        fig.write_image(pdf_filename)

        # Show interactive heatmap in a notebook or webpage
        fig.show()

        return pdf_filename

    async def execute(self):
        await self.ts_client.connect()
        available_pairs = await self.ts_client.get_available_pairs()
        table_names = [self.ts_client.get_trades_table_name(connector_name, trading_pair)
                       for connector_name, trading_pair in available_pairs]
        await self.set_base_metrics()

        active_bots_status = await self.backend_api_client.get_active_bots_status()
        performance_data = active_bots_status["data"]
        performance_by_bot_dict = {instance_name: data.get("performance") for instance_name, data in performance_data.items()}
        performance_string = json.dumps(performance_by_bot_dict, indent=4)
        # Generate the heatmap PDF
        # heatmap_pdf = await self.generate_heatmap()

        # Generate the report and prepare the email
        report, csv_dict = self.generate_report(table_names, performance_string)

        message = self.create_email(
            subject="Database Refresh Report - Thinking Science Journal",
            sender_email=self.config["email"],
            recipients=self.config["recipients"],
            body=report
        )

        # Attach CSV files, heatmap PDF, and other files
        # for filename in ["all_daily_metrics.csv", heatmap_pdf] + [f"{k}.csv" for k in csv_dict]:
        # TODO: try to replace all_daily_metrics by a class attribute like self.daily_metrics
        for filename in ["all_daily_metrics.csv"] + [f"{k}.csv" for k in csv_dict]:
            try:
                self.add_attachment(message, filename)
            except Exception as e:
                print(f"Unable to attach file {filename}: {e}")

        # Send the email
        self.send_email(message, sender_email=self.config["email"], app_password=self.config["email_password"])

    def generate_report(self, table_names: List[str], bots_report: str = None) -> (str, Dict[str, pd.DataFrame]):
        final_df = self.base_metrics
        missing_pairs_list = [pair for pair in table_names if pair not in final_df['table_names'].unique()]
        outdated_pairs_list = [pair for pair in table_names if
                               pair in final_df[final_df['when'] == 'yesterday']['table_names'].unique()]
        correct_pairs_list = [pair for pair in table_names if
                              pair in final_df[final_df['when'] == 'today']['table_names'].unique()]
        new_pairs_list = [pair for pair in table_names if
                          pair in final_df[final_df['is_new']]['table_names'].unique()]

        report = f"\n\nHello Mr Pickantell!:\n"
        report += f"Here are your fucking bots running:\n"
        if len(bots_report) < 10:
            report += f"You have no fucking bots\n"
        else:
            report += str(bots_report) + "\n\n"
        report += f"\nHere's a quick review on your database:\n"
        if not missing_pairs_list and not outdated_pairs_list:
            report += "\n--> All trading pairs have been updated today."
        if missing_pairs_list:
            if len(missing_pairs_list) > 20:
                report += "\n\n--> Missing trading pairs (not updated in 2 days):\nToo many pairs, printing missing_pairs.csv file instead"
            else:
                report += "\n\n--> Missing trading pairs (not updated in 2 days):\n" + "\n".join(
                    f" - {pair}" for pair in missing_pairs_list)
        if outdated_pairs_list:
            if len(outdated_pairs_list) > 20:
                report += "\n\n--> Outdated trading pairs (not updated since yesterday):\nToo many pairs, printing outdated_pairs.csv file instead"
            else:
                report += "\n\n--> Outdated trading pairs (not updated since yesterday):\n" + "\n".join(
                    f" - {pair}" for pair in outdated_pairs_list)
        if correct_pairs_list:
            if len(correct_pairs_list) > 20:
                report += "\n\n--> Correct trading pairs (up to date):\nToo many pairs, printing correct_pairs.csv file instead"
            else:
                report += "\n\n--> Correct trading pairs (up to date):\n" + "\n".join(
                    f" - {pair}" for pair in correct_pairs_list)
        if new_pairs_list:
            if len(new_pairs_list) > 20:
                report += "\n\n--> New trading pairs (up to date):\nToo many pairs, printing new_pairs.csv file instead"
            else:
                report += "\n\n--> New trading pairs (up to date):\n" + "\n".join(
                    f" - {pair}" for pair in new_pairs_list)
        report += f"\n\nAdditional Database Flux Information:\n\n"

        report += f"--> Amount of trading pairs missing (no info for 2 days) out of total pairs:{(len(missing_pairs_list)/len(table_names)):.2f}%\n"
        report += f"--> Outdated pairs (no info since yesterday) out of total pairs: {(len(outdated_pairs_list)/len(table_names)):.2f}%\n"
        report += f"--> Correct pairs (updated info) out of total pairs: {(len(correct_pairs_list) / len(table_names)):.2f}%\n"
        report += f"--> New pairs out of total pairs:: {(len(new_pairs_list) / len(table_names)):.2f}%\n\n"

        report += f"\n\nFor more information visit the attached files:\n++ all_daily_metrics.csv: general information about current databases\n++ trade_amount_heatmap.pdf: Per trading pair - % of total trades downloaded per day\n\n"
        report += "See you soon and don't forget to be awesome!!"

        csv_dict = {
            "missing_pairs": final_df[final_df['trading_pair'].isin(missing_pairs_list)],
            "outdated_pairs": final_df[final_df['trading_pair'].isin(outdated_pairs_list)],
            "correct_pairs": final_df[final_df['trading_pair'].isin(correct_pairs_list)],
            "new_pairs": final_df[final_df['trading_pair'].isin(new_pairs_list)]
        }
        return report, {key: df for key, df in csv_dict.items() if len(df) > 20}


async def main():
    config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "backend_api_host": os.getenv("TRADING_HOST", "localhost"),
        "email": "thinkingscience.ts@gmail.com",
        "email_password": os.getenv("EMAIL_PASSWORD", "password"),
        "recipients": os.getenv("RECIPIENTS", "").split(","),
        "export": True
    }
    task = ReportGeneratorTask(name="Report Generator", frequency=timedelta(hours=12), config=config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
