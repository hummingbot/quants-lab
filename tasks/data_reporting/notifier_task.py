from data_reporting_task import TaskBase
from datetime import datetime, timedelta
from core.services.backend_api_client import BackendAPIClient

class WarningNotifier(TaskBase):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.report = None
        self.base_metrics = None

    async def get_base_tables(self):
        available_pairs = await self.ts_client.get_available_pairs()
        table_names = [self.ts_client.get_trades_table_name(connector_name, trading_pair)
                       for connector_name, trading_pair in available_pairs]
        return table_names

    async def set_base_metrics(self):
        self.base_metrics = await self.ts_client.get_db_status_df()
        self.base_metrics["table_names"] = self.base_metrics.apply(lambda x: self.ts_client.get_trades_table_name(x["connector_name"], x["trading_pair"]), axis=1)

    async def execute(self):
        self.report = "\nHi Mr Pickantell, we regret to tell you some bad news :-( \n"
        await self.ts_client.connect()
        available_pairs = await self.ts_client.get_available_pairs()
        table_names = [self.ts_client.get_trades_table_name(connector_name, trading_pair)
                       for connector_name, trading_pair in available_pairs]
        await self.set_base_metrics()
        self.base_metrics.sort_values(['to_timestamp'], ascending = False, inplace = True)
        try:
            max_timestamp = self.base_metrics['to_timestamp'][0]
            now = datetime.now()
            # Check if the difference is greater than 1 day
            if now - max_timestamp > timedelta(days=1):
                self.report += f"\nLast Trading Pair Summary Update was in {max_timestamp}, so it is outdated!\n"
                table_name = self.ts_client.get_trades_table_name(self.base_metrics["connector_name"][0], self.base_metrics["trading_pair"][0])
                max_timestamp = await execute_query(query = f"select max(date(timestamp)) max_timestamp from {table_name}")
                if now - max_timestamp > timedelta(days=1):
                    self.report += f"\nTrading pair {table_name} taken as sample: \nMax timestamp: {max_timestamp}, so it is outdated!\n"
                else:
                    self.report += f"\nHowever, {table_name} taken as sample: \nMax timestamp: {max_timestamp}, so last info was less than 1d before!\n"
        except:
            self.base_metrics['to_timestamp'] = np.nan

        active_bots_status = await self.backend_api_client.get_active_bots_status()
        performance_data = active_bots_status["data"]
        performance_by_bot_dict = {instance_name: data.get("performance") for instance_name, data in performance_data.items()}
        performance_string = json.dumps(performance_by_bot_dict, indent=4)
        if len(performance_string) > 0:
            self.report += "\nAt least these bots are running"
            self.report += performance_string
        else:
            self.report += "\nAlso, you have no fucking bots running!! Aaaaalgo el frendo"
        message = self.create_email(
            subject="Database Refresh Report - Thinking Science Journal",
            sender_email=self.config["email"],
            recipients=self.config["recipients"],
            body=self.report
        )
        # Send the email
        self.send_email(message, sender_email=self.config["email"], app_password=self.config["email_password"])


async def main():
    config = {
        "host": os.getenv("TIMESCALE_HOST", "63.250.52.93"),
        "backend_api_host": os.getenv("TRADING_HOST", "63.250.52.93"),
        "email": "thinkingscience.ts@gmail.com",
        "email_password": "dqtn zjkf aumv esak",
        # "recipients": ["palmiscianoblas@gmail.com", "federico.cardoso.e@gmail.com", "apelsantiago@gmail.com",  "tomasgaudino8@gmail.com"]
        "recipients": ["palmiscianoblas@gmail.com"],
        "export": True,
    }
    task = WarningNotifier(name="Report Generator", frequency=timedelta(hours=12), config=config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
