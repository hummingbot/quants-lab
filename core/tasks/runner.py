"""
Enhanced Task Runner with new task system integration.
"""
import asyncio
import importlib
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import uvicorn
import yaml
from dotenv import load_dotenv

from core.tasks import MongoDBTaskStorage
from core.tasks.base import BaseTask, TaskConfig, ScheduleConfig
from core.tasks.orchestrator import TaskOrchestrator
from core.tasks.api import app, set_orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskRunner:
    """
    Enhanced task runner with new task system.
    """
    
    def __init__(self, config_path: str = "config/tasks.yml", enable_api: bool = None):
        self.config_path = config_path
        self.orchestrator: Optional[TaskOrchestrator] = None
        self.api_server: Optional[uvicorn.Server] = None
        
        # API configuration with fallback to environment variables
        if enable_api is not None:
            self.api_enabled = enable_api
        else:
            self.api_enabled = os.getenv("TASK_API_ENABLED", "false").lower() == "true"
        
        self.api_host = os.getenv("TASK_API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("TASK_API_PORT", "8000"))
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {self.config_path}")
        return config
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            if self.orchestrator:
                asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _check_mongodb_configured(self) -> None:
        """Check if MongoDB is configured in environment."""
        mongo_uri = os.getenv("MONGO_URI")
        mongo_database = os.getenv("MONGO_DATABASE", "quants_lab")
        
        # Debug logging to check environment variables
        logger.info("=== MongoDB Environment Check ===")
        logger.info(f"MONGO_URI: {'Configured' if mongo_uri else 'Not configured'}")
        logger.info(f"MONGO_DATABASE: {mongo_database}")
        logger.info("=================================")
        
        if not mongo_uri:
            logger.warning("MONGO_URI not set in environment. Storage will not be available.")
    
    def _create_task_config(self, task_name: str, task_data: Dict[str, Any]) -> TaskConfig:
        """Create TaskConfig from configuration data."""
        # Parse schedule configuration
        schedule_data = task_data.get("schedule", {})
        
        # Handle legacy frequency_hours
        if "frequency_hours" in task_data:
            schedule_data = {
                "type": "frequency",
                "frequency_hours": task_data["frequency_hours"]
            }
        
        schedule = ScheduleConfig(**schedule_data)
        
        # Create task configuration
        config = TaskConfig(
            name=task_name,
            enabled=task_data.get("enabled", True),
            task_class=task_data["task_class"],
            schedule=schedule,
            max_retries=task_data.get("max_retries", 3),
            retry_delay_seconds=task_data.get("retry_delay_seconds", 60),
            timeout_seconds=task_data.get("timeout_seconds"),
            dependencies=task_data.get("dependencies", []),
            config=task_data.get("config", {}),
            tags=task_data.get("tags", [])
        )
        
        return config
    
    def _import_task_class(self, task_class_path: str) -> type:
        """Import task class from string path."""
        try:
            # Resolve simplified task class names
            from core.tasks.registry import resolve_task_class
            resolved_path = resolve_task_class(task_class_path)
            
            module_path, class_name = resolved_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            logger.error(f"Error importing task class {task_class_path} (resolved to {resolved_path}): {e}")
            raise
    
    def _create_task_instance(self, config: TaskConfig) -> BaseTask:
        """Create task instance from configuration."""
        task_class = self._import_task_class(config.task_class)
        return task_class(config)
    
    async def _initialize_tasks(self) -> List[BaseTask]:
        """Initialize all tasks from configuration."""
        tasks = []
        
        if "tasks" not in self.config:
            logger.warning("No tasks found in configuration")
            return tasks
        
        for task_name, task_data in self.config["tasks"].items():
            try:
                logger.info(f"Initializing task: {task_name}")
                
                # Create task configuration
                config = self._create_task_config(task_name, task_data)
                
                # Create task instance
                task = self._create_task_instance(config)
                
                tasks.append(task)
                logger.info(f"Successfully initialized task: {task_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize task {task_name}: {e}")
                if self.config.get("strict_mode", False):
                    raise
                continue
        
        return tasks
    
    async def _start_api_server(self):
        """Start the FastAPI server if enabled."""
        if not self.api_enabled:
            logger.info("API server disabled")
            return
        
        logger.info(f"Starting API server on {self.api_host}:{self.api_port}")
        
        # Set orchestrator for API
        set_orchestrator(self.orchestrator)
        
        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=self.api_host,
            port=self.api_port,
            log_level="info",
            loop="asyncio"
        )
        
        self.api_server = uvicorn.Server(config)
        
        # Start server in background
        await self.api_server.serve()
    
    async def start(self):
        """Start the task runner."""
        logger.info("Starting QuantsLab Task Runner v2.0")
        
        try:
            # Check MongoDB configuration
            self._check_mongodb_configured()
            
            # Initialize storage (no config needed, uses db_manager)
            storage = MongoDBTaskStorage()
            
            # Create orchestrator
            max_concurrent = self.config.get("max_concurrent_tasks", 10)
            self.orchestrator = TaskOrchestrator(
                storage=storage,
                max_concurrent_tasks=max_concurrent,
                retry_failed_tasks=self.config.get("retry_failed_tasks", True)
            )
            
            # Initialize tasks
            tasks = await self._initialize_tasks()
            
            if not tasks:
                logger.warning("No tasks initialized, exiting...")
                return
            
            # Add tasks to orchestrator
            for task in tasks:
                self.orchestrator.add_task(task)
            
            logger.info(f"Initialized {len(tasks)} tasks")
            
            # Start API server if enabled
            if self.api_enabled:
                api_task = asyncio.create_task(self._start_api_server())
            else:
                api_task = None
            
            # Start orchestrator
            orchestrator_task = asyncio.create_task(self.orchestrator.start())
            
            # Wait for tasks to complete
            tasks_to_wait = [orchestrator_task]
            if api_task:
                tasks_to_wait.append(api_task)
            
            await asyncio.gather(*tasks_to_wait)
            
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Error starting task runner: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the task runner gracefully."""
        logger.info("Stopping task runner...")
        
        # Stop orchestrator
        if self.orchestrator:
            await self.orchestrator.stop()
        
        # Stop API server
        if self.api_server:
            logger.info("Stopping API server...")
            self.api_server.should_exit = True
        
        logger.info("Task runner stopped")
    
    async def reload_config(self, new_config_path: Optional[str] = None):
        """Reload configuration and restart tasks."""
        logger.info("Reloading configuration...")
        
        if new_config_path:
            self.config_path = new_config_path
        
        # Load new configuration
        new_config = self._load_config()
        
        # Update tasks in orchestrator
        if self.orchestrator:
            # Remove old tasks
            old_task_names = set(self.orchestrator.tasks.keys())
            
            # Initialize new tasks
            new_tasks = await self._initialize_tasks()
            new_task_names = {task.config.name for task in new_tasks}
            
            # Remove tasks that are no longer in config
            for task_name in old_task_names - new_task_names:
                self.orchestrator.remove_task(task_name)
                logger.info(f"Removed task: {task_name}")
            
            # Add or update tasks
            for task in new_tasks:
                task_name = task.config.name
                if task_name in old_task_names:
                    # Reload existing task
                    await self.orchestrator.reload_task(task_name, task.config)
                    logger.info(f"Reloaded task: {task_name}")
                else:
                    # Add new task
                    self.orchestrator.add_task(task)
                    logger.info(f"Added new task: {task_name}")
        
        self.config = new_config
        logger.info("Configuration reloaded successfully")
    
    def load_config(self) -> Dict[str, Any]:
        """Public method to access configuration data for CLI commands."""
        return self.config
    
    def validate_config(self) -> bool:
        """Validate the current configuration."""
        try:
            logger.info("Validating configuration...")
            
            if "tasks" not in self.config:
                logger.error("No 'tasks' section found in configuration")
                return False
            
            for task_name, task_data in self.config["tasks"].items():
                try:
                    # Validate task configuration by creating TaskConfig
                    config = self._create_task_config(task_name, task_data)
                    
                    # Validate task class can be imported
                    self._import_task_class(config.task_class)
                    
                    logger.info(f"Task {task_name}: OK")
                    
                except Exception as e:
                    logger.error(f"Task {task_name}: INVALID - {e}")
                    return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


async def main():
    """Main entry point for the task runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantsLab Task Runner v2.0")
    parser.add_argument(
        "--config",
        default="config/tasks.yml",
        help="Path to task configuration file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration and exit"
    )
    parser.add_argument(
        "--api-host",
        default="0.0.0.0",
        help="API server host"
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port"
    )
    parser.add_argument(
        "--disable-api",
        action="store_true",
        help="Disable API server"
    )
    
    args = parser.parse_args()
    
    # Override environment variables with command line arguments
    if args.disable_api:
        os.environ["TASK_API_ENABLED"] = "false"
    else:
        os.environ["TASK_API_HOST"] = args.api_host
        os.environ["TASK_API_PORT"] = str(args.api_port)
    
    # Create task runner
    runner = TaskRunner(config_path=args.config)
    
    # Validate configuration
    if not runner.validate_config():
        logger.error("Configuration validation failed, exiting...")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration is valid")
        return
    
    # Start task runner
    await runner.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Task runner interrupted by user")
    except Exception as e:
        logger.error(f"Task runner failed: {e}")
        sys.exit(1)