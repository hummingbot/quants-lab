"""
Enhanced Task Orchestrator with dependency management and task chaining.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from collections import defaultdict
import pytz

from core.tasks.base import BaseTask, TaskConfig, TaskContext, TaskResult, TaskStatus
from core.tasks.storage import TaskStorage, TimescaleDBTaskStorage

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """
    Task orchestrator that manages task execution, dependencies, and scheduling.
    """
    
    def __init__(
        self,
        storage: TaskStorage,
        max_concurrent_tasks: int = 10,
        retry_failed_tasks: bool = True
    ):
        self.storage = storage
        self.max_concurrent_tasks = max_concurrent_tasks
        self.retry_failed_tasks = retry_failed_tasks
        
        self.tasks: Dict[str, BaseTask] = {}
        self.task_configs: Dict[str, TaskConfig] = {}
        self.running_tasks: Set[str] = set()
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Dependency tracking
        self.task_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.dependency_triggers: Dict[str, List[TaskConfig]] = defaultdict(list)
        
        # Control flags
        self._running = False
        self._stop_event = asyncio.Event()
        
    def add_task(self, task: BaseTask) -> None:
        """
        Add a task to the orchestrator.
        
        Args:
            task: Task instance to add
        """
        config = task.config
        self.tasks[config.name] = task
        self.task_configs[config.name] = config
        
        # Setup dependencies
        for dep in config.dependencies:
            self.dependency_triggers[dep.task_name].append(config)
            self.task_dependencies[config.name].append(dep.task_name)
        
        logger.info(f"Added task: {config.name} with {len(config.dependencies)} dependencies")
    
    def remove_task(self, task_name: str) -> None:
        """
        Remove a task from the orchestrator.
        
        Args:
            task_name: Name of task to remove
        """
        if task_name in self.tasks:
            del self.tasks[task_name]
            del self.task_configs[task_name]
            
            # Clean up dependencies
            for deps in self.dependency_triggers.values():
                deps[:] = [d for d in deps if d.name != task_name]
            
            if task_name in self.task_dependencies:
                del self.task_dependencies[task_name]
            
            logger.info(f"Removed task: {task_name}")
    
    async def execute_task(
        self,
        task_name: str,
        context: Optional[TaskContext] = None,
        force: bool = False
    ) -> Optional[TaskResult]:
        """
        Execute a single task.
        
        Args:
            task_name: Name of task to execute
            context: Optional task context
            force: Force execution even if already running
            
        Returns:
            Task execution result or None if skipped
        """
        if task_name not in self.tasks:
            logger.error(f"Task {task_name} not found")
            return None
        
        task = self.tasks[task_name]
        config = self.task_configs[task_name]
        
        # Check if task is already running
        if not force and task_name in self.running_tasks:
            logger.warning(f"Task {task_name} is already running, skipping")
            return None
        
        # Check if task should run based on schedule
        if not force and context is None:
            last_execution = await self.storage.get_last_execution(task_name)
            last_run = last_execution.started_at if last_execution else None
            
            if not task.should_run_now(last_run):
                logger.debug(f"Task {task_name} not scheduled to run yet")
                return None
        
        # Acquire semaphore for concurrency control
        async with self.task_semaphore:
            # Mark task as running
            self.running_tasks.add(task_name)
            
            # Try to mark in storage (prevents duplicate runs across instances)
            if not await self.storage.mark_task_running(task_name, context.execution_id if context else ""):
                logger.warning(f"Task {task_name} is already running in another instance")
                self.running_tasks.discard(task_name)
                return None
            
            try:
                # Create context if not provided
                if context is None:
                    context = TaskContext(task_name=task_name)
                
                # Execute task with retries
                result = None
                last_error = None
                
                for attempt in range(1, config.max_retries + 1):
                    context.attempt_number = attempt
                    
                    try:
                        result = await task.run(context)
                        
                        if result.status == TaskStatus.COMPLETED:
                            break
                        elif result.status == TaskStatus.FAILED and attempt < config.max_retries:
                            logger.warning(f"Task {task_name} failed on attempt {attempt}, retrying...")
                            await task.on_retry(context, attempt, Exception(result.error_message))
                            await asyncio.sleep(config.retry_delay_seconds)
                        
                    except Exception as e:
                        last_error = e
                        logger.error(f"Error executing task {task_name} on attempt {attempt}: {e}")
                        
                        if attempt < config.max_retries:
                            await task.on_retry(context, attempt, e)
                            await asyncio.sleep(config.retry_delay_seconds)
                        else:
                            # Create failed result
                            result = TaskResult(
                                execution_id=context.execution_id,
                                task_name=task_name,
                                status=TaskStatus.FAILED,
                                started_at=context.started_at,
                                completed_at=datetime.now(timezone.utc),
                                error_message=str(e)
                            )
                            result.calculate_duration()
                
                # Save execution result
                if result:
                    await self.storage.save_execution(result, context)
                    
                    # Trigger dependent tasks
                    await self._trigger_dependencies(task_name, result, context)
                
                return result
                
            finally:
                # Mark task as no longer running
                self.running_tasks.discard(task_name)
                await self.storage.mark_task_completed(task_name)
    
    async def _trigger_dependencies(
        self,
        task_name: str,
        result: TaskResult,
        parent_context: TaskContext
    ) -> None:
        """
        Trigger dependent tasks based on execution result.
        
        Args:
            task_name: Name of completed task
            result: Task execution result
            parent_context: Parent task context
        """
        if task_name not in self.dependency_triggers:
            return
        
        for dep_config in self.dependency_triggers[task_name]:
            should_trigger = False
            
            # Find the specific dependency configuration
            dep_def = next(
                (d for d in dep_config.dependencies if d.task_name == task_name),
                None
            )
            
            if not dep_def:
                continue
            
            # Check trigger conditions
            if dep_def.on_completion:
                should_trigger = True
            elif dep_def.on_success and result.status == TaskStatus.COMPLETED:
                should_trigger = True
            elif dep_def.on_failure and result.status == TaskStatus.FAILED:
                should_trigger = True
            
            if should_trigger:
                # Wait for delay if specified
                if dep_def.delay_seconds > 0:
                    await asyncio.sleep(dep_def.delay_seconds)
                
                # Create child context
                child_context = TaskContext(
                    task_name=dep_config.name,
                    triggered_by=f"dependency:{task_name}",
                    parent_execution_id=parent_context.execution_id
                )
                
                logger.info(f"Triggering dependent task {dep_config.name} after {task_name}")
                
                # Execute dependent task asynchronously
                asyncio.create_task(self.execute_task(dep_config.name, child_context))
    
    async def _schedule_loop(self) -> None:
        """Main scheduling loop."""
        logger.info("Starting task scheduler loop")
        
        while self._running:
            try:
                # Get all enabled tasks
                for task_name, task in self.tasks.items():
                    if not task.config.enabled:
                        continue
                    
                    # Skip if task has dependencies (will be triggered by parent)
                    if task_name in self.task_dependencies and self.task_dependencies[task_name]:
                        continue
                    
                    # Check if task should run
                    last_execution = await self.storage.get_last_execution(task_name)
                    last_run = last_execution.started_at if last_execution else None
                    
                    if task.should_run_now(last_run):
                        # Execute task asynchronously
                        asyncio.create_task(self.execute_task(task_name))
                
                # Wait before next scheduling check
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(10)
    
    async def start(self) -> None:
        """Start the orchestrator."""
        if self._running:
            logger.warning("Orchestrator is already running")
            return
        
        logger.info(f"Starting orchestrator with {len(self.tasks)} tasks")
        
        # Initialize storage
        await self.storage.initialize()
        
        self._running = True
        self._stop_event.clear()
        
        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._schedule_loop())
        
        # Wait for stop signal
        await self._stop_event.wait()
    
    async def stop(self) -> None:
        """Stop the orchestrator."""
        if not self._running:
            logger.warning("Orchestrator is not running")
            return
        
        logger.info("Stopping orchestrator...")
        
        self._running = False
        self._stop_event.set()
        
        # Cancel scheduler task
        if hasattr(self, '_scheduler_task'):
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for running tasks to complete
        while self.running_tasks:
            logger.info(f"Waiting for {len(self.running_tasks)} tasks to complete...")
            await asyncio.sleep(1)
        
        # Close storage
        await self.storage.close()
        
        logger.info("Orchestrator stopped")
    
    async def trigger_task(
        self,
        task_name: str,
        triggered_by: str = "manual",
        metadata: Optional[Dict] = None
    ) -> Optional[TaskResult]:
        """
        Manually trigger a task execution.
        
        Args:
            task_name: Name of task to trigger
            triggered_by: Source of trigger
            metadata: Optional metadata for execution
            
        Returns:
            Task execution result
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")
        
        context = TaskContext(
            task_name=task_name,
            triggered_by=triggered_by,
            metadata=metadata or {}
        )
        
        return await self.execute_task(task_name, context, force=True)
    
    def get_task_status(self, task_name: str) -> Dict:
        """
        Get current status of a task.
        
        Args:
            task_name: Name of task
            
        Returns:
            Task status information
        """
        if task_name not in self.tasks:
            return {"error": f"Task {task_name} not found"}
        
        task = self.tasks[task_name]
        config = self.task_configs[task_name]
        
        return {
            "name": task_name,
            "enabled": config.enabled,
            "is_running": task_name in self.running_tasks,
            "schedule": {
                "type": config.schedule.type,
                "frequency_hours": config.schedule.frequency_hours,
                "cron": config.schedule.cron,
                "timezone": config.schedule.timezone
            },
            "dependencies": [
                {
                    "task_name": dep.task_name,
                    "on_success": dep.on_success,
                    "on_failure": dep.on_failure,
                    "on_completion": dep.on_completion
                }
                for dep in config.dependencies
            ],
            "next_run": task.get_next_run_time().isoformat() if task.get_next_run_time() else None
        }
    
    def get_all_tasks_status(self) -> List[Dict]:
        """Get status of all tasks."""
        return [self.get_task_status(task_name) for task_name in self.tasks.keys()]
    
    async def reload_task(self, task_name: str, new_config: TaskConfig) -> None:
        """
        Reload a task with new configuration.
        
        Args:
            task_name: Name of task to reload
            new_config: New task configuration
        """
        # Remove old task
        if task_name in self.tasks:
            self.remove_task(task_name)
        
        # Import and create new task instance
        import importlib
        module_path, class_name = new_config.task_class.rsplit('.', 1)
        module = importlib.import_module(module_path)
        task_class = getattr(module, class_name)
        
        # Create new task
        new_task = task_class(new_config)
        
        # Add new task
        self.add_task(new_task)
        
        logger.info(f"Reloaded task: {task_name}")
    
    async def pause_task(self, task_name: str) -> None:
        """Pause a task (disable it)."""
        if task_name in self.task_configs:
            self.task_configs[task_name].enabled = False
            logger.info(f"Paused task: {task_name}")
    
    async def resume_task(self, task_name: str) -> None:
        """Resume a paused task (enable it)."""
        if task_name in self.task_configs:
            self.task_configs[task_name].enabled = True
            logger.info(f"Resumed task: {task_name}")
    
    async def get_execution_history(
        self,
        task_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get task execution history.
        
        Args:
            task_name: Optional task name filter
            limit: Maximum number of records
            
        Returns:
            List of execution records
        """
        executions = await self.storage.get_executions(
            task_name=task_name,
            limit=limit
        )
        
        return [
            {
                "execution_id": e.execution_id,
                "task_name": e.task_name,
                "status": e.status,
                "started_at": e.started_at.isoformat(),
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "duration_seconds": e.duration_seconds,
                "triggered_by": e.triggered_by,
                "error_message": e.error_message
            }
            for e in executions
        ]