import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
from email.mime.base import MIMEBase
from email import encoders

from core.data_sources import CLOBDataSource
from core.services import TimescaleClient
from core.task_base import BaseTask

from reporter_test import get_base, get_final_df, print_str, get_part_attach

logging.basicConfig(level=logging.INFO)


class ReportGeneratorTask(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)

    async def execute(self):
        base = get_base()
        trading_pairs = base[base['table_name'].str.contains('_trades')]['table_name'].unique()
        final_df = get_final_df(trading_pairs)

        try:
            # Email configuration
            sender_email = 'thinkingscience.ts@gmail.com'
            # recipients = ['palmiscianoblas@gmail.com']
            recipients = ['palmiscianoblas@gmail.com','federico.cardoso.e@gmail.com', 'tomasgaudino8@gmail.com', 'apelsantiago@gmail.com']
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
            logging.error("An error occurred: while running Report Generator%s", e)
            print(f"Error: {e}")

        # Clean up
        connection.close()
        engine.dispose()



if __name__ == "__main__":

    task = ReportGeneratorTask("Report Generator", timedelta(hours=12))
    asyncio.run(task.execute())
