import asyncio
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict
import logging
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTask(ABC):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        self.name = name
        self.frequency = frequency
        self.config = config
        self.last_run = None
        self.logs = []
        self.metadata = {
            "name": self.name,
            "timestamp": self.now(),
            "server": "localhost",
            "owner": "admin",
            "frequency": self.frequency.seconds,
            "config": self.config,
            "logs": self.logs,
        }

    @abstractmethod
    async def execute(self):
        pass

    def reset_metadata(self):
        self.metadata = {
            "name": self.name,
            "timestamp": self.now(),
            "server": "localhost",
            "owner": "admin",
            "frequency": self.frequency.seconds,
            "config": self.config,
            "logs": self.logs,
        }

    async def run_with_frequency(self):
        while True:
            now = datetime.now()
            if self.last_run is None or (now - self.last_run) >= self.frequency:
                try:
                    self.last_run = now
                    await self.execute()
                except Exception as e:
                    logger.info(f" Error executing task {self.name}: {e}")
            await asyncio.sleep(1)  # Check every second

    @staticmethod
    def now():
        return pd.to_datetime(time.time(), unit="s").strftime("%Y-%m-%d %H:%M:%S")


class TaskOrchestrator:
    def __init__(self):
        self.tasks = []

    def add_task(self, task: BaseTask):
        self.tasks.append(task)

    async def run(self):
        task_coroutines = [task.run_with_frequency() for task in self.tasks]
        await asyncio.gather(*task_coroutines)
