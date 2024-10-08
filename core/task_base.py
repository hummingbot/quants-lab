import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict


class BaseTask(ABC):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        self.name = name
        self.frequency = frequency
        self.config = config
        self.last_run = None

    @abstractmethod
    async def execute(self):
        pass

    async def run_with_frequency(self):
        while True:
            now = datetime.now()
            if self.last_run is None or (now - self.last_run) >= self.frequency:
                try:
                    self.last_run = now
                    await self.execute()
                except Exception as e:
                    print(f"Error executing task {self.name}: {e}")
            await asyncio.sleep(1)  # Check every second


class TaskOrchestrator:
    def __init__(self):
        self.tasks = []

    def add_task(self, task: BaseTask):
        self.tasks.append(task)

    async def run(self):
        task_coroutines = [task.run_with_frequency() for task in self.tasks]
        await asyncio.gather(*task_coroutines)
