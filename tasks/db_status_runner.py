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
    from tasks.data_collection.db_status_task import DbStatusTask
    orchestrator = TaskOrchestrator()

    config = {
        'db_host': os.getenv("TIMESCALE_HOST", 'localhost'),
        'db_port': 5432,
    }

    db_status_task = DbStatusTask("Metrics Report", timedelta(hours=1), config)
    orchestrator.add_task(db_status_task)

    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(main())
