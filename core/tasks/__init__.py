"""
QuantsLab Core Task System v2.0

This module provides the core infrastructure for task management, scheduling,
and orchestration in the QuantsLab trading system.

Main Components:
- BaseTask: Enhanced base class with lifecycle management and Pydantic validation
- TaskConfig: Configuration models with cron scheduling support
- TaskOrchestrator: Manages task execution, dependencies, and scheduling
- TimescaleDBTaskStorage: Time-series storage for task execution history
- TaskRunner: Main application runner with API integration
- FastAPI: REST API endpoints for external task control

Usage:
    from core.tasks import BaseTask, TaskConfig, TaskContext
    from core.tasks.runner import TaskRunner
    from core.tasks.orchestrator import TaskOrchestrator
    
    # Create custom task
    class MyTask(BaseTask):
        async def execute(self, context: TaskContext):
            return {"status": "success"}
    
    # Run task system
    runner = TaskRunner(config_path="config/tasks.yml")
    await runner.start()
"""

# Import main components for easier access
from .base import (
    BaseTask,
    TaskConfig,
    TaskContext,
    TaskResult,
    TaskStatus,
    ScheduleConfig,
    TaskDependency
)

from .storage import (
    TaskStorage,
    MongoDBTaskStorage,
    TaskExecutionRecord
)

from .orchestrator import TaskOrchestrator

from .runner import TaskRunner

__all__ = [
    # Base classes and models
    'BaseTask',
    'TaskConfig',
    'TaskContext',
    'TaskResult',
    'TaskStatus',
    'ScheduleConfig',
    'TaskDependency',
    
    # Storage
    'TaskStorage',
    'MongoDBTaskStorage',
    'TaskExecutionRecord',
    
    # Orchestration
    'TaskOrchestrator',
    
    # Runner
    'TaskRunner'
]

# Version info
__version__ = '2.0.0'