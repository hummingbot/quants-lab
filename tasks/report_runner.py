import asyncio
import logging
import os
import sys
from datetime import timedelta

from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    from core.task_base import TaskOrchestrator
    from tasks.data_reporting.data_reporting_task import ReportGeneratorTask
    from tasks.data_reporting.notifier_task import WarningNotifier

    orchestrator = TaskOrchestrator()
    config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "backend_api_host": os.getenv("TRADING_HOST", "localhost"),
        "email": os.getenv("EMAIL_SENDER"),
        "email_password": os.getenv("EMAIL_PASSWORD"),
        "recipients": os.getenv("EMAIL_RECIPIENTS").split(","),
        "export": True
    }

    data_reporting_task = ReportGeneratorTask("Screener Report", timedelta(hours=8), config)
    warning_task = WarningNotifier("Warning Task", timedelta(2), config=config)

    orchestrator.add_task(data_reporting_task)
    orchestrator.add_task(warning_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
