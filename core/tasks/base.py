"""
Enhanced Task Base Module with Pydantic 2.0 validation and improved lifecycle management.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from enum import Enum
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from croniter import croniter
import pytz

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class ScheduleConfig(BaseModel):
    """Schedule configuration for tasks."""
    model_config = ConfigDict(extra='forbid')
    
    type: str = Field(default="frequency", description="Schedule type: frequency or cron")
    frequency_hours: Optional[float] = Field(None, description="Frequency in hours")
    cron: Optional[str] = Field(None, description="Cron expression")
    timezone: str = Field(default="UTC", description="Timezone for cron schedules")
    start_time: Optional[datetime] = Field(None, description="Start time for the schedule")
    end_time: Optional[datetime] = Field(None, description="End time for the schedule")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ["frequency", "cron"]:
            raise ValueError("Schedule type must be 'frequency' or 'cron'")
        return v
    
    @field_validator('cron')
    @classmethod
    def validate_cron(cls, v: Optional[str], info) -> Optional[str]:
        if info.data.get('type') == 'cron' and v:
            if not croniter.is_valid(v):
                raise ValueError(f"Invalid cron expression: {v}")
        return v
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        try:
            pytz.timezone(v)
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(f"Unknown timezone: {v}")
        return v


class TaskDependency(BaseModel):
    """Task dependency configuration."""
    model_config = ConfigDict(extra='forbid')
    
    task_name: str = Field(..., description="Name of the dependent task")
    on_success: bool = Field(default=False, description="Trigger on success")
    on_failure: bool = Field(default=False, description="Trigger on failure")
    on_completion: bool = Field(default=False, description="Trigger on any completion")
    delay_seconds: int = Field(default=0, description="Delay before triggering")


class TaskConfig(BaseModel):
    """Complete task configuration."""
    model_config = ConfigDict(extra='allow')  # Allow extra fields for simplified config
    
    name: str = Field(..., description="Task name")
    enabled: bool = Field(default=True, description="Whether task is enabled")
    task_class: str = Field(..., description="Python path to task class or simplified name")
    schedule: ScheduleConfig = Field(..., description="Schedule configuration")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay_seconds: int = Field(default=60, description="Delay between retries")
    timeout_seconds: Optional[int] = Field(None, description="Task execution timeout")
    dependencies: List[TaskDependency] = Field(default_factory=list, description="Task dependencies")
    config: Dict[str, Any] = Field(default_factory=dict, description="Task-specific configuration")
    tags: List[str] = Field(default_factory=list, description="Task tags for filtering")
    
    @field_validator('task_class')
    @classmethod
    def validate_task_class(cls, v: str) -> str:
        if not v:
            raise ValueError("Task class cannot be empty")
        # Allow simplified names (like 'notebook') or full paths
        return v
    
    @model_validator(mode='after')
    def move_simplified_config_to_config(self):
        """Move simplified config fields to the config dict for backwards compatibility."""
        # Only process notebook tasks
        if self.task_class in ['notebook', 'notebooks'] or 'notebook' in self.task_class.lower():
            # Fields that should be moved to the config dict for notebook tasks
            simplified_fields = [
                'notebooks', 'execution', 'parameters', 'timeout', 
                'max_parallel', 'continue_on_error', 'kernel'
            ]
            
            # Move simplified fields to config dict
            extra_data = getattr(self, '__pydantic_extra__', {})
            for field in simplified_fields:
                if field in extra_data and field not in self.config:
                    self.config[field] = extra_data[field]
        
        return self


class TaskContext(BaseModel):
    """Runtime context for task execution."""
    model_config = ConfigDict(extra='forbid')
    
    execution_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attempt_number: int = Field(default=1)
    triggered_by: str = Field(default="scheduler")
    parent_execution_id: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    """Result of task execution."""
    model_config = ConfigDict(extra='forbid')
    
    execution_id: str
    task_name: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    def calculate_duration(self) -> None:
        """Calculate task duration."""
        if self.completed_at and self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class BaseTask(ABC):
    """Enhanced base task with Pydantic validation and lifecycle management."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.context: Optional[TaskContext] = None
        self.result: Optional[TaskResult] = None
        self._is_running = False
        self._lock = asyncio.Lock()
        
        # Database clients (will be initialized if needed)
        self.mongodb_client = None
        
    @abstractmethod
    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """
        Execute the task logic.
        
        Args:
            context: Task execution context
            
        Returns:
            Dictionary containing task results
        """
        pass
    
    async def validate_prerequisites(self) -> bool:
        """
        Validate task prerequisites before execution.
        Override in subclasses for custom validation.
        
        Returns:
            True if prerequisites are met, False otherwise
        """
        return True
    
    async def setup(self, context: TaskContext) -> None:
        """
        Setup task before execution.
        Override in subclasses for custom setup.
        
        Args:
            context: Task execution context
        """
        # Initialize databases if requested in task config
        task_config = self.config.config
        
        if task_config.get('use_mongodb', False):
            await self._init_mongodb()

    async def _init_mongodb(self) -> None:
        """Initialize MongoDB client."""
        try:
            from core.database_manager import db_manager
            self.mongodb_client = await db_manager.get_mongodb_client()
        except Exception as e:
            logger.warning(f"Failed to initialize MongoDB: {e}")

    async def cleanup(self, context: TaskContext, result: TaskResult) -> None:
        """
        Cleanup after task execution.
        Override in subclasses for custom cleanup.
        
        Args:
            context: Task execution context
            result: Task execution result
        """
        pass
    
    async def on_success(self, context: TaskContext, result: TaskResult) -> None:
        """
        Hook called on successful task completion.
        Override in subclasses for custom success handling.
        
        Args:
            context: Task execution context
            result: Task execution result
        """
        pass
    
    async def on_failure(self, context: TaskContext, result: TaskResult) -> None:
        """
        Hook called on task failure.
        Override in subclasses for custom failure handling.
        
        Args:
            context: Task execution context
            result: Task execution result
        """
        pass
    
    async def on_retry(self, context: TaskContext, attempt: int, error: Exception) -> None:
        """
        Hook called before retry attempt.
        Override in subclasses for custom retry handling.
        
        Args:
            context: Task execution context
            attempt: Retry attempt number
            error: Exception that caused the retry
        """
        pass
    
    async def run(self, context: Optional[TaskContext] = None) -> TaskResult:
        """
        Run the task with complete lifecycle management.
        
        Args:
            context: Optional task context, will be created if not provided
            
        Returns:
            Task execution result
        """
        async with self._lock:
            if self._is_running:
                logger.warning(f"Task {self.config.name} is already running, skipping")
                return TaskResult(
                    execution_id=str(uuid.uuid4()),
                    task_name=self.config.name,
                    status=TaskStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc),
                    error_message="Task is already running"
                )
            
            self._is_running = True
        
        # Create context if not provided
        if context is None:
            context = TaskContext(task_name=self.config.name)
        
        self.context = context
        
        # Initialize result
        result = TaskResult(
            execution_id=context.execution_id,
            task_name=self.config.name,
            status=TaskStatus.RUNNING,
            started_at=context.started_at
        )
        
        try:
            # Validate prerequisites
            if not await self.validate_prerequisites():
                raise RuntimeError("Task prerequisites not met")
            
            # Setup
            await self.setup(context)
            
            # Execute with timeout if specified
            if self.config.timeout_seconds:
                result_data = await asyncio.wait_for(
                    self.execute(context),
                    timeout=self.config.timeout_seconds
                )
            else:
                result_data = await self.execute(context)
            
            # Update result
            result.status = TaskStatus.COMPLETED
            result.result_data = result_data
            result.completed_at = datetime.now(timezone.utc)
            result.calculate_duration()
            
            # Call success hook
            await self.on_success(context, result)
            
        except asyncio.TimeoutError as e:
            logger.error(f"Task {self.config.name} timed out after {self.config.timeout_seconds} seconds")
            result.status = TaskStatus.FAILED
            result.error_message = f"Task timed out after {self.config.timeout_seconds} seconds"
            result.completed_at = datetime.now(timezone.utc)
            result.calculate_duration()
            
            # Call failure hook
            await self.on_failure(context, result)
            
        except Exception as e:
            logger.error(f"Task {self.config.name} failed: {str(e)}")
            result.status = TaskStatus.FAILED
            result.error_message = str(e)
            result.completed_at = datetime.now(timezone.utc)
            result.calculate_duration()
            
            # Store traceback
            import traceback
            result.error_traceback = traceback.format_exc()
            
            # Call failure hook
            await self.on_failure(context, result)
            
        finally:
            # Cleanup
            await self.cleanup(context, result)
            
            # Reset running flag
            self._is_running = False
            
            # Store result
            self.result = result
        
        return result
    
    def get_next_run_time(self) -> Optional[datetime]:
        """
        Calculate the next run time based on schedule configuration.
        
        Returns:
            Next run time or None if task is disabled
        """
        if not self.config.enabled:
            return None
        
        now = datetime.now(pytz.UTC)
        
        # Check schedule window
        if self.config.schedule.start_time and now < self.config.schedule.start_time:
            return self.config.schedule.start_time
        
        if self.config.schedule.end_time and now > self.config.schedule.end_time:
            return None
        
        # Calculate based on schedule type
        if self.config.schedule.type == "cron" and self.config.schedule.cron:
            tz = pytz.timezone(self.config.schedule.timezone)
            cron = croniter(self.config.schedule.cron, now.astimezone(tz))
            return cron.get_next(datetime)
        
        elif self.config.schedule.type == "frequency" and self.config.schedule.frequency_hours:
            # For frequency-based, return current time + frequency
            from datetime import timedelta
            return now + timedelta(hours=self.config.schedule.frequency_hours)
        
        return None
    
    def should_run_now(self, last_run: Optional[datetime] = None) -> bool:
        """
        Check if task should run now based on schedule.
        
        Args:
            last_run: Last execution time
            
        Returns:
            True if task should run now
        """
        if not self.config.enabled:
            return False
        
        now = datetime.now(pytz.UTC)
        
        # Check schedule window
        if self.config.schedule.start_time and now < self.config.schedule.start_time:
            return False
        
        if self.config.schedule.end_time and now > self.config.schedule.end_time:
            return False
        
        # Check based on schedule type
        if self.config.schedule.type == "cron" and self.config.schedule.cron:
            if last_run is None:
                return True
            
            tz = pytz.timezone(self.config.schedule.timezone)
            cron = croniter(self.config.schedule.cron, last_run.astimezone(tz))
            next_run = cron.get_next(datetime)
            return now >= next_run
        
        elif self.config.schedule.type == "frequency" and self.config.schedule.frequency_hours:
            if last_run is None:
                return True
            
            from datetime import timedelta
            return now >= last_run + timedelta(hours=self.config.schedule.frequency_hours)
        
        return False