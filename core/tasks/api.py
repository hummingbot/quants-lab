"""
FastAPI endpoints for task management and external triggers.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.tasks.orchestrator import TaskOrchestrator
from core.tasks.base import TaskStatus, TaskContext
import uuid

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="QuantsLab Task Management API",
    description="API for managing and triggering trading tasks",
    version="2.0.0"
)

# Global orchestrator instance (will be set by TaskRunner)
orchestrator: Optional[TaskOrchestrator] = None


def get_orchestrator() -> TaskOrchestrator:
    """Dependency to get orchestrator instance."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Task orchestrator not initialized")
    return orchestrator


# Request/Response models
class TriggerTaskRequest(BaseModel):
    """Request model for triggering a task."""
    task_name: str = Field(..., description="Name of task to trigger")
    triggered_by: str = Field(default="api", description="Source of trigger")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    force: bool = Field(default=True, description="Force execution even if already running")


class TriggerTaskResponse(BaseModel):
    """Response model for task trigger."""
    execution_id: str
    task_name: str
    status: str
    message: str
    started_at: datetime


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    name: str
    enabled: bool
    is_running: bool
    schedule: Dict[str, Any]
    dependencies: List[Dict[str, Any]]
    next_run: Optional[str]


class ExecutionHistoryResponse(BaseModel):
    """Response model for execution history."""
    execution_id: str
    task_name: str
    status: str
    started_at: str
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    triggered_by: str
    error_message: Optional[str]


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    time_bucket: datetime
    task_name: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_duration: Optional[float]
    median_duration: Optional[float]
    p95_duration: Optional[float]
    max_duration: Optional[float]
    min_duration: Optional[float]


# Helper functions for background task execution
async def _execute_task_background(
    orch: TaskOrchestrator,
    task_name: str,
    triggered_by: str,
    metadata: dict,
    execution_id: str
):
    """Execute task in background without blocking the API response."""
    try:
        logger.info(f"Starting background execution of task {task_name} (execution_id: {execution_id})")
        
        # Create context with the provided execution_id
        context = TaskContext(
            execution_id=execution_id,
            task_name=task_name,
            triggered_by=triggered_by,
            metadata=metadata
        )
        
        # Execute the task through orchestrator
        await orch.execute_task(
            task_name=task_name,
            context=context,
            force=True  # Force execution even if already running
        )
        
        logger.info(f"Background task {task_name} completed (execution_id: {execution_id})")
        
    except Exception as e:
        logger.error(f"Background task {task_name} failed (execution_id: {execution_id}): {e}")
        # The error will be stored in the task storage by the orchestrator


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "orchestrator_initialized": orchestrator is not None
    }


# Task management endpoints
@app.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    enabled_only: bool = Query(False, description="Filter to only enabled tasks"),
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """List all registered tasks."""
    tasks = orch.get_all_tasks_status()
    
    if enabled_only:
        tasks = [t for t in tasks if t["enabled"]]
    
    return tasks


@app.get("/tasks/{task_name}", response_model=TaskStatusResponse)
async def get_task_status(
    task_name: str,
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Get status of a specific task."""
    status = orch.get_task_status(task_name)
    
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    
    return status


@app.post("/tasks/{task_name}/trigger", response_model=TriggerTaskResponse)
async def trigger_task(
    task_name: str,
    request: TriggerTaskRequest,
    background_tasks: BackgroundTasks,
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Manually trigger a task execution."""
    try:
        # Validate task exists
        if task_name not in orch.tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_name} not found")
        
        # Check if task is already running (optional)
        if not request.force and task_name in orch.running_tasks:
            raise HTTPException(
                status_code=409,
                detail=f"Task {task_name} is already running. Set force=true to queue another execution."
            )
        
        # Generate execution ID for tracking
        execution_id = str(uuid.uuid4())
        
        # Add task to background execution
        background_tasks.add_task(
            _execute_task_background,
            orch,
            task_name,
            request.triggered_by,
            request.metadata,
            execution_id
        )
        
        # Return immediately with pending status
        return TriggerTaskResponse(
            execution_id=execution_id,
            task_name=task_name,
            status=TaskStatus.PENDING.value,
            message=f"Task {task_name} queued for execution",
            started_at=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering task {task_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/tasks/{task_name}/pause")
async def pause_task(
    task_name: str,
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Pause (disable) a task."""
    if task_name not in orch.tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_name} not found")
    
    await orch.pause_task(task_name)
    
    return {"message": f"Task {task_name} paused successfully"}


@app.post("/tasks/{task_name}/resume")
async def resume_task(
    task_name: str,
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Resume (enable) a paused task."""
    if task_name not in orch.tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_name} not found")
    
    await orch.resume_task(task_name)
    
    return {"message": f"Task {task_name} resumed successfully"}


# Execution history endpoints
@app.get("/executions", response_model=List[ExecutionHistoryResponse])
async def get_execution_history(
    task_name: Optional[str] = Query(None, description="Filter by task name"),
    status: Optional[TaskStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Get task execution history."""
    executions = await orch.storage.get_executions(
        task_name=task_name,
        status=status,
        limit=limit
    )
    
    return [
        ExecutionHistoryResponse(
            execution_id=e.execution_id,
            task_name=e.task_name,
            status=e.status,
            started_at=e.started_at.isoformat(),
            completed_at=e.completed_at.isoformat() if e.completed_at else None,
            duration_seconds=e.duration_seconds,
            triggered_by=e.triggered_by,
            error_message=e.error_message
        )
        for e in executions
    ]


@app.get("/executions/{execution_id}")
async def get_execution_detail(
    execution_id: str,
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Get detailed information about a specific execution."""
    # Try to get from storage
    try:
        executions = await orch.storage.get_executions(limit=100)
        execution = next((e for e in executions if e.execution_id == execution_id), None)
        
        if not execution:
            # Check if it's currently running
            if execution_id in [str(getattr(task, 'context', {}).get('execution_id', '')) 
                                for task in orch.running_tasks.values() if hasattr(task, 'context')]:
                return {
                    "execution_id": execution_id,
                    "status": TaskStatus.RUNNING.value,
                    "message": "Task is currently running"
                }
            
            raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
        
        return {
            "execution_id": execution.execution_id,
            "task_name": execution.task_name,
            "status": execution.status,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "duration_seconds": execution.duration_seconds,
            "triggered_by": execution.triggered_by,
            "attempt_number": execution.attempt_number,
            "parent_execution_id": execution.parent_execution_id,
            "result_data": execution.result_data,
            "error_message": execution.error_message,
            "error_traceback": execution.error_traceback,
            "metrics": execution.metrics,
            "metadata": execution.metadata
        }
    except AttributeError:
        # Storage might not be initialized
        return {
            "execution_id": execution_id,
            "status": "unknown",
            "message": "Storage backend not available"
        }


# Performance metrics endpoints
@app.get("/metrics/performance", response_model=List[PerformanceMetricsResponse])
async def get_performance_metrics(
    task_name: Optional[str] = Query(None, description="Filter by task name"),
    interval: str = Query("1 hour", description="Time bucket interval (e.g., '1 hour', '1 day')"),
    last_days: int = Query(7, ge=1, le=90, description="Number of days to look back"),
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Get aggregated performance metrics for tasks."""
    if not hasattr(orch.storage, 'get_task_performance_metrics'):
        raise HTTPException(
            status_code=501,
            detail="Performance metrics not supported by current storage backend"
        )
    
    metrics = await orch.storage.get_task_performance_metrics(
        task_name=task_name,
        interval=interval,
        last_days=last_days
    )
    
    return [
        PerformanceMetricsResponse(**metric)
        for metric in metrics
    ]


@app.get("/metrics/anomalies")
async def get_anomalies(
    task_name: str,
    threshold_stddev: float = Query(3.0, description="Standard deviation threshold for anomaly detection"),
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Detect anomalous task executions based on duration."""
    if not hasattr(orch.storage, 'detect_anomalies'):
        raise HTTPException(
            status_code=501,
            detail="Anomaly detection not supported by current storage backend"
        )
    
    anomalies = await orch.storage.detect_anomalies(
        task_name=task_name,
        threshold_stddev=threshold_stddev
    )
    
    return anomalies


@app.get("/metrics/running")
async def get_running_tasks(
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Get list of currently running tasks."""
    running_tasks = list(orch.running_tasks)
    
    if hasattr(orch.storage, 'get_running_tasks'):
        storage_running = await orch.storage.get_running_tasks()
        running_tasks = list(set(running_tasks + storage_running))
    
    return {
        "running_tasks": running_tasks,
        "count": len(running_tasks)
    }


# Batch operations
@app.post("/tasks/trigger-batch")
async def trigger_batch_tasks(
    task_names: List[str],
    background_tasks: BackgroundTasks,
    triggered_by: str = "api-batch",
    force: bool = True,
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Trigger multiple tasks in batch."""
    results = []
    
    for task_name in task_names:
        try:
            # Validate task exists
            if task_name not in orch.tasks:
                results.append({
                    "task_name": task_name,
                    "status": "failed",
                    "error": f"Task {task_name} not found"
                })
                continue
            
            # Check if already running (unless force is set)
            if not force and task_name in orch.running_tasks:
                results.append({
                    "task_name": task_name,
                    "status": "skipped",
                    "error": "Task is already running"
                })
                continue
            
            # Generate execution ID for tracking
            execution_id = str(uuid.uuid4())
            
            # Add task to background execution
            background_tasks.add_task(
                _execute_task_background,
                orch,
                task_name,
                triggered_by,
                {},  # Empty metadata for batch triggers
                execution_id
            )
            
            results.append({
                "task_name": task_name,
                "status": "queued",
                "execution_id": execution_id
            })
            
        except Exception as e:
            results.append({
                "task_name": task_name,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "triggered_count": sum(1 for r in results if r["status"] == "queued"),
        "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
        "failed_count": sum(1 for r in results if r["status"] == "failed"),
        "results": results
    }


@app.post("/tasks/pause-all")
async def pause_all_tasks(
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Pause all tasks."""
    paused = []
    
    for task_name in orch.tasks.keys():
        await orch.pause_task(task_name)
        paused.append(task_name)
    
    return {
        "message": f"Paused {len(paused)} tasks",
        "tasks": paused
    }


@app.post("/tasks/resume-all")
async def resume_all_tasks(
    orch: TaskOrchestrator = Depends(get_orchestrator)
):
    """Resume all tasks."""
    resumed = []
    
    for task_name in orch.tasks.keys():
        await orch.resume_task(task_name)
        resumed.append(task_name)
    
    return {
        "message": f"Resumed {len(resumed)} tasks",
        "tasks": resumed
    }


# WebSocket endpoint for real-time updates (optional)
from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)
        
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()


@app.websocket("/ws/tasks")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time task updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic updates
            if orchestrator:
                status = {
                    "type": "status_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "running_tasks": list(orchestrator.running_tasks),
                    "total_tasks": len(orchestrator.tasks)
                }
                await websocket.send_json(status)
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Utility function to set orchestrator (called by TaskRunner)
def set_orchestrator(orch: TaskOrchestrator):
    """Set the global orchestrator instance."""
    global orchestrator
    orchestrator = orch
    logger.info("Task orchestrator set for API")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )