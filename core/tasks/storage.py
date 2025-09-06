"""
TimescaleDB Storage Implementation for Task Management System.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import asyncio
import json

import asyncpg
from asyncpg.pool import Pool
from pydantic import BaseModel, Field

from core.tasks.base import TaskResult, TaskStatus, TaskContext, TaskConfig

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


class TimescaleDBTaskStorage(TaskStorage):
    """TimescaleDB implementation of task storage."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[Pool] = None
        
    async def initialize(self) -> None:
        """Initialize TimescaleDB connection pool and create tables."""
        # Create connection pool
        self.pool = await asyncpg.create_pool(
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 5432),
            user=self.config.get("user", "admin"),
            password=self.config.get("password", "admin"),
            database=self.config.get("database", "timescaledb"),
            min_size=2,
            max_size=10
        )
        
        # Create tables and hypertables
        async with self.pool.acquire() as conn:
            # Create TimescaleDB extension if not exists
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            
            # Create task_executions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_executions (
                    execution_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ,
                    duration_seconds DOUBLE PRECISION,
                    triggered_by TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    parent_execution_id TEXT,
                    result_data JSONB,
                    error_message TEXT,
                    error_traceback TEXT,
                    metrics JSONB,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create hypertable for time-series data
            await conn.execute("""
                SELECT create_hypertable('task_executions', 'started_at',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 day'
                );
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_executions_task_name 
                ON task_executions (task_name, started_at DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_executions_status 
                ON task_executions (status, started_at DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_executions_parent 
                ON task_executions (parent_execution_id) 
                WHERE parent_execution_id IS NOT NULL;
            """)
            
            # Create task_schedules table for tracking last runs
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS task_schedules (
                    task_name TEXT PRIMARY KEY,
                    last_run TIMESTAMPTZ,
                    next_run TIMESTAMPTZ,
                    is_running BOOLEAN DEFAULT FALSE,
                    current_execution_id TEXT,
                    run_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    average_duration_seconds DOUBLE PRECISION,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create compression policy (compress data older than 1 day)
            await conn.execute("""
                SELECT add_compression_policy('task_executions', 
                    INTERVAL '1 day',
                    if_not_exists => TRUE
                );
            """)
            
            # Create retention policy (drop data older than 90 days)
            await conn.execute("""
                SELECT add_retention_policy('task_executions', 
                    INTERVAL '90 days',
                    if_not_exists => TRUE
                );
            """)
            
            # Create continuous aggregate for hourly stats
            await conn.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS task_hourly_stats
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('1 hour', started_at) AS hour,
                    task_name,
                    COUNT(*) as execution_count,
                    COUNT(*) FILTER (WHERE status = 'completed') as success_count,
                    COUNT(*) FILTER (WHERE status = 'failed') as failure_count,
                    AVG(duration_seconds) as avg_duration,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_seconds) as median_duration,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_seconds) as p95_duration,
                    MAX(duration_seconds) as max_duration,
                    MIN(duration_seconds) as min_duration
                FROM task_executions
                GROUP BY hour, task_name
                WITH NO DATA;
            """)
            
            # Refresh policy for continuous aggregate
            await conn.execute("""
                SELECT add_continuous_aggregate_policy('task_hourly_stats',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """)
            
        logger.info("TimescaleDB storage initialized successfully")
    
    async def close(self) -> None:
        """Close TimescaleDB connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def save_execution(self, result: TaskResult, context: TaskContext) -> None:
        """Save task execution result to TimescaleDB."""
        async with self.pool.acquire() as conn:
            # Save execution record
            await conn.execute("""
                INSERT INTO task_executions (
                    execution_id, task_name, status, started_at, completed_at,
                    duration_seconds, triggered_by, attempt_number, parent_execution_id,
                    result_data, error_message, error_traceback, metrics, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            """,
                result.execution_id,
                result.task_name,
                result.status.value,
                result.started_at,
                result.completed_at,
                result.duration_seconds,
                context.triggered_by,
                context.attempt_number,
                context.parent_execution_id,
                json.dumps(result.result_data) if result.result_data else None,
                result.error_message,
                result.error_traceback,
                json.dumps(result.metrics) if result.metrics else None,
                json.dumps(context.metadata) if context.metadata else None
            )
            
            # Update task schedule
            await conn.execute("""
                INSERT INTO task_schedules (
                    task_name, last_run, is_running, current_execution_id, 
                    run_count, success_count, failure_count
                ) VALUES ($1, $2, $3, $4, 1, $5, $6)
                ON CONFLICT (task_name) DO UPDATE SET
                    last_run = EXCLUDED.last_run,
                    is_running = EXCLUDED.is_running,
                    current_execution_id = EXCLUDED.current_execution_id,
                    run_count = task_schedules.run_count + 1,
                    success_count = task_schedules.success_count + $5,
                    failure_count = task_schedules.failure_count + $6,
                    updated_at = NOW()
            """,
                result.task_name,
                result.started_at,
                result.status == TaskStatus.RUNNING,
                result.execution_id if result.status == TaskStatus.RUNNING else None,
                1 if result.status == TaskStatus.COMPLETED else 0,
                1 if result.status == TaskStatus.FAILED else 0
            )
            
            # Update average duration if completed
            if result.status == TaskStatus.COMPLETED and result.duration_seconds:
                await conn.execute("""
                    UPDATE task_schedules SET
                        average_duration_seconds = (
                            SELECT AVG(duration_seconds) 
                            FROM task_executions 
                            WHERE task_name = $1 
                            AND status = 'completed' 
                            AND started_at > NOW() - INTERVAL '7 days'
                        )
                    WHERE task_name = $1
                """, result.task_name)
    
    async def get_last_execution(self, task_name: str) -> Optional[TaskExecutionRecord]:
        """Get last execution record for a task."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM task_executions
                WHERE task_name = $1
                ORDER BY started_at DESC
                LIMIT 1
            """, task_name)
            
            if row:
                return TaskExecutionRecord(
                    execution_id=row["execution_id"],
                    task_name=row["task_name"],
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    duration_seconds=row["duration_seconds"],
                    triggered_by=row["triggered_by"],
                    attempt_number=row["attempt_number"],
                    parent_execution_id=row["parent_execution_id"],
                    result_data=json.loads(row["result_data"]) if row["result_data"] else None,
                    error_message=row["error_message"],
                    error_traceback=row["error_traceback"],
                    metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
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
        query = "SELECT * FROM task_executions WHERE 1=1"
        params = []
        param_count = 0
        
        if task_name:
            param_count += 1
            query += f" AND task_name = ${param_count}"
            params.append(task_name)
        
        if status:
            param_count += 1
            query += f" AND status = ${param_count}"
            params.append(status.value)
        
        if start_time:
            param_count += 1
            query += f" AND started_at >= ${param_count}"
            params.append(start_time)
        
        if end_time:
            param_count += 1
            query += f" AND started_at <= ${param_count}"
            params.append(end_time)
        
        query += f" ORDER BY started_at DESC LIMIT {limit}"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            
            return [
                TaskExecutionRecord(
                    execution_id=row["execution_id"],
                    task_name=row["task_name"],
                    status=row["status"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    duration_seconds=row["duration_seconds"],
                    triggered_by=row["triggered_by"],
                    attempt_number=row["attempt_number"],
                    parent_execution_id=row["parent_execution_id"],
                    result_data=json.loads(row["result_data"]) if row["result_data"] else None,
                    error_message=row["error_message"],
                    error_traceback=row["error_traceback"],
                    metrics=json.loads(row["metrics"]) if row["metrics"] else {},
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
                for row in rows
            ]
    
    async def get_task_performance_metrics(
        self, 
        task_name: Optional[str] = None,
        interval: str = '1 hour',
        last_days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get aggregated performance metrics for tasks."""
        query = """
            SELECT 
                time_bucket($1, started_at) AS time_bucket,
                task_name,
                COUNT(*) as total_executions,
                COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
                AVG(duration_seconds) as avg_duration,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_seconds) as median_duration,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_seconds) as p95_duration,
                MAX(duration_seconds) as max_duration,
                MIN(duration_seconds) as min_duration
            FROM task_executions 
            WHERE started_at >= NOW() - $2::INTERVAL
        """
        
        params = [interval, f"{last_days} days"]
        
        if task_name:
            query += " AND task_name = $3"
            params.append(task_name)
        
        query += " GROUP BY time_bucket, task_name ORDER BY time_bucket DESC"
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [dict(row) for row in rows]
    
    async def detect_anomalies(self, task_name: str, threshold_stddev: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalous task executions based on duration."""
        query = """
            WITH task_stats AS (
                SELECT 
                    AVG(duration_seconds) as mean_duration,
                    STDDEV(duration_seconds) as stddev_duration
                FROM task_executions
                WHERE task_name = $1
                AND status = 'completed'
                AND started_at >= NOW() - INTERVAL '30 days'
            )
            SELECT 
                e.execution_id,
                e.started_at,
                e.duration_seconds,
                s.mean_duration,
                s.stddev_duration,
                ABS(e.duration_seconds - s.mean_duration) / NULLIF(s.stddev_duration, 0) as z_score
            FROM task_executions e
            CROSS JOIN task_stats s
            WHERE e.task_name = $1
            AND e.status = 'completed'
            AND e.started_at >= NOW() - INTERVAL '7 days'
            AND ABS(e.duration_seconds - s.mean_duration) > $2 * s.stddev_duration
            ORDER BY e.started_at DESC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, task_name, threshold_stddev)
            return [dict(row) for row in rows]
    
    async def get_task_dependencies_execution(
        self, 
        parent_execution_id: str
    ) -> List[TaskExecutionRecord]:
        """Get all child task executions for a parent execution."""
        return await self.get_executions(
            parent_execution_id=parent_execution_id
        )
    
    async def mark_task_running(self, task_name: str, execution_id: str) -> bool:
        """Mark a task as running (for preventing duplicate runs)."""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                UPDATE task_schedules 
                SET is_running = TRUE, 
                    current_execution_id = $2,
                    updated_at = NOW()
                WHERE task_name = $1 AND is_running = FALSE
                RETURNING TRUE
            """, task_name, execution_id)
            return bool(result)
    
    async def mark_task_completed(self, task_name: str) -> None:
        """Mark a task as completed (no longer running)."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE task_schedules 
                SET is_running = FALSE, 
                    current_execution_id = NULL,
                    updated_at = NOW()
                WHERE task_name = $1
            """, task_name)
    
    async def get_running_tasks(self) -> List[str]:
        """Get list of currently running tasks."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT task_name FROM task_schedules WHERE is_running = TRUE
            """)
            return [row["task_name"] for row in rows]