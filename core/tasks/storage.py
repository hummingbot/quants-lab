"""
MongoDB Storage Implementation for Task Management System.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field
from pymongo import DESCENDING, ASCENDING

from core.tasks.base import TaskResult, TaskStatus, TaskContext

logger = logging.getLogger(__name__)


class TaskExecutionRecord(BaseModel):
    """Record of a task execution in the database."""
    execution_id: str
    task_name: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    triggered_by: str
    attempt_number: int
    parent_execution_id: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskStorage(ABC):
    """Abstract base class for task storage."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize storage backend."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""
        pass
    
    @abstractmethod
    async def save_execution(self, result: TaskResult, context: TaskContext) -> None:
        """Save task execution result."""
        pass
    
    @abstractmethod
    async def get_last_execution(self, task_name: str) -> Optional[TaskExecutionRecord]:
        """Get last execution record for a task."""
        pass
    
    @abstractmethod
    async def get_executions(
        self,
        task_name: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TaskExecutionRecord]:
        """Get task executions with filters."""
        pass


class MongoDBTaskStorage(TaskStorage):
    """MongoDB implementation of task storage."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.executions_collection = None
        self.schedules_collection = None
        
    async def initialize(self) -> None:
        """Initialize MongoDB connection and create indexes."""
        from core.database_manager import db_manager
        import os
        
        # Use centralized database manager
        mongodb_client = await db_manager.get_mongodb_client()
        if mongodb_client is None:
            raise RuntimeError("Failed to get MongoDB client from database manager. Please ensure MONGO_URI is set in environment variables.")
        
        # Get database name from environment
        database = os.getenv('MONGO_DATABASE', 'quants_lab')
        
        # Debug logging for MongoDB connection
        logger.info("=== MongoDB Storage Initialization ===")
        logger.info(f"Using centralized database manager")
        logger.info(f"Database: {database}")
        logger.info("=======================================")
        
        # Use the centralized client and get database
        self.client = mongodb_client.client  # Access underlying AsyncIOMotorClient
        self.db = mongodb_client.get_database(database)
        
        # Get collections
        self.executions_collection = self.db["task_executions"]
        self.schedules_collection = self.db["task_schedules"]
        
        # Create indexes for task_executions
        await self.executions_collection.create_index([("task_name", ASCENDING), ("started_at", DESCENDING)])
        await self.executions_collection.create_index([("status", ASCENDING), ("started_at", DESCENDING)])
        await self.executions_collection.create_index([("execution_id", ASCENDING)], unique=True)
        await self.executions_collection.create_index([("parent_execution_id", ASCENDING)], sparse=True)
        await self.executions_collection.create_index([("started_at", DESCENDING)])
        
        # Create TTL index to automatically delete old records (90 days)
        await self.executions_collection.create_index(
            [("created_at", ASCENDING)],
            expireAfterSeconds=90 * 24 * 60 * 60  # 90 days in seconds
        )
        
        # Create indexes for task_schedules
        await self.schedules_collection.create_index([("task_name", ASCENDING)], unique=True)
        await self.schedules_collection.create_index([("is_running", ASCENDING)])
        
        logger.info("MongoDB storage initialized successfully")
    
    async def close(self) -> None:
        """Close MongoDB connection."""
        # Don't close the shared connection from database_manager
        # The database_manager handles its own lifecycle
        self.client = None
        self.db = None
    
    async def save_execution(self, result: TaskResult, context: TaskContext) -> None:
        """Save task execution result to MongoDB."""
        # Prepare execution record
        execution_doc = {
            "execution_id": result.execution_id,
            "task_name": result.task_name,
            "status": result.status.value,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "duration_seconds": result.duration_seconds,
            "triggered_by": context.triggered_by,
            "attempt_number": context.attempt_number,
            "parent_execution_id": context.parent_execution_id,
            "result_data": result.result_data,
            "error_message": result.error_message,
            "error_traceback": result.error_traceback,
            "metrics": result.metrics,
            "metadata": context.metadata,
            "created_at": datetime.utcnow()
        }
        
        # Save execution record
        await self.executions_collection.insert_one(execution_doc)
        
        # Update task schedule
        schedule_update = {
            "$set": {
                "last_run": result.started_at,
                "is_running": result.status == TaskStatus.RUNNING,
                "current_execution_id": result.execution_id if result.status == TaskStatus.RUNNING else None,
                "updated_at": datetime.utcnow()
            },
            "$inc": {
                "run_count": 1,
                "success_count": 1 if result.status == TaskStatus.COMPLETED else 0,
                "failure_count": 1 if result.status == TaskStatus.FAILED else 0
            }
        }
        
        # Update average duration if completed
        if result.status == TaskStatus.COMPLETED and result.duration_seconds:
            # Calculate new average duration
            schedule_doc = await self.schedules_collection.find_one({"task_name": result.task_name})
            if schedule_doc:
                current_avg = schedule_doc.get("average_duration_seconds", 0)
                current_count = schedule_doc.get("success_count", 0)
                new_avg = ((current_avg * current_count) + result.duration_seconds) / (current_count + 1)
                schedule_update["$set"]["average_duration_seconds"] = new_avg
            else:
                schedule_update["$set"]["average_duration_seconds"] = result.duration_seconds
        
        await self.schedules_collection.update_one(
            {"task_name": result.task_name},
            schedule_update,
            upsert=True
        )
    
    async def get_last_execution(self, task_name: str) -> Optional[TaskExecutionRecord]:
        """Get last execution record for a task."""
        doc = await self.executions_collection.find_one(
            {"task_name": task_name},
            sort=[("started_at", DESCENDING)]
        )
        
        if doc:
            return TaskExecutionRecord(
                execution_id=doc["execution_id"],
                task_name=doc["task_name"],
                status=doc["status"],
                started_at=doc["started_at"],
                completed_at=doc.get("completed_at"),
                duration_seconds=doc.get("duration_seconds"),
                triggered_by=doc["triggered_by"],
                attempt_number=doc["attempt_number"],
                parent_execution_id=doc.get("parent_execution_id"),
                result_data=doc.get("result_data"),
                error_message=doc.get("error_message"),
                error_traceback=doc.get("error_traceback"),
                metrics=doc.get("metrics", {}),
                metadata=doc.get("metadata", {})
            )
        return None
    
    async def get_executions(
        self,
        task_name: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TaskExecutionRecord]:
        """Get task executions with filters."""
        query = {}
        
        if task_name:
            query["task_name"] = task_name
        
        if status:
            query["status"] = status.value
        
        if start_time or end_time:
            query["started_at"] = {}
            if start_time:
                query["started_at"]["$gte"] = start_time
            if end_time:
                query["started_at"]["$lte"] = end_time
        
        cursor = self.executions_collection.find(query).sort("started_at", DESCENDING).limit(limit)
        
        executions = []
        async for doc in cursor:
            executions.append(TaskExecutionRecord(
                execution_id=doc["execution_id"],
                task_name=doc["task_name"],
                status=doc["status"],
                started_at=doc["started_at"],
                completed_at=doc.get("completed_at"),
                duration_seconds=doc.get("duration_seconds"),
                triggered_by=doc["triggered_by"],
                attempt_number=doc["attempt_number"],
                parent_execution_id=doc.get("parent_execution_id"),
                result_data=doc.get("result_data"),
                error_message=doc.get("error_message"),
                error_traceback=doc.get("error_traceback"),
                metrics=doc.get("metrics", {}),
                metadata=doc.get("metadata", {})
            ))
        
        return executions
    
    async def get_task_performance_metrics(
        self, 
        task_name: Optional[str] = None,
        last_days: int = 7
    ) -> Dict[str, Any]:
        """Get aggregated performance metrics for tasks."""
        pipeline = []
        
        # Filter by date range
        start_date = datetime.utcnow() - timedelta(days=last_days)
        pipeline.append({
            "$match": {
                "started_at": {"$gte": start_date}
            }
        })
        
        # Filter by task name if provided
        if task_name:
            pipeline[0]["$match"]["task_name"] = task_name
        
        # Group and calculate metrics
        pipeline.extend([
            {
                "$group": {
                    "_id": "$task_name",
                    "total_executions": {"$sum": 1},
                    "successful_executions": {
                        "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                    },
                    "failed_executions": {
                        "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                    },
                    "avg_duration": {"$avg": "$duration_seconds"},
                    "max_duration": {"$max": "$duration_seconds"},
                    "min_duration": {"$min": "$duration_seconds"},
                    "durations": {"$push": "$duration_seconds"}
                }
            },
            {
                "$project": {
                    "task_name": "$_id",
                    "total_executions": 1,
                    "successful_executions": 1,
                    "failed_executions": 1,
                    "success_rate": {
                        "$multiply": [
                            {"$divide": ["$successful_executions", "$total_executions"]},
                            100
                        ]
                    },
                    "avg_duration": {"$round": ["$avg_duration", 2]},
                    "max_duration": 1,
                    "min_duration": 1,
                    "_id": 0
                }
            }
        ])
        
        cursor = self.executions_collection.aggregate(pipeline)
        metrics = []
        async for doc in cursor:
            metrics.append(doc)
        
        return {
            "period_days": last_days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.utcnow().isoformat(),
            "metrics": metrics
        }
    
    async def mark_task_running(self, task_name: str, execution_id: str) -> bool:
        """Mark a task as running (for preventing duplicate runs)."""
        # First, try to find and update if exists and not running
        result = await self.schedules_collection.update_one(
            {"task_name": task_name, "is_running": False},
            {
                "$set": {
                    "is_running": True,
                    "current_execution_id": execution_id,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # If no document was modified, check if it exists
        if result.modified_count == 0:
            # Check if task exists and is already running
            existing = await self.schedules_collection.find_one({"task_name": task_name})
            if existing and existing.get("is_running", False):
                return False  # Task is already running
            
            # If document doesn't exist, create it
            if not existing:
                await self.schedules_collection.insert_one({
                    "task_name": task_name,
                    "is_running": True,
                    "current_execution_id": execution_id,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                })
                return True
        
        return result.modified_count > 0
    
    async def mark_task_completed(self, task_name: str) -> None:
        """Mark a task as completed (no longer running)."""
        await self.schedules_collection.update_one(
            {"task_name": task_name},
            {
                "$set": {
                    "is_running": False,
                    "current_execution_id": None,
                    "updated_at": datetime.utcnow()
                }
            }
        )
    
    async def get_running_tasks(self) -> List[str]:
        """Get list of currently running tasks."""
        cursor = self.schedules_collection.find({"is_running": True})
        tasks = []
        async for doc in cursor:
            tasks.append(doc["task_name"])
        return tasks
    
    async def get_task_schedule(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get schedule information for a task."""
        return await self.schedules_collection.find_one({"task_name": task_name})