#!/usr/bin/env python3
"""
QuantsLab CLI - Main entry point for task management
"""
import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from core.tasks.runner import TaskRunner
from core.tasks.base import TaskConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='QuantsLab Task Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tasks continuously from config
  python cli.py run-tasks --config config/data_collection_v2.yml
  
  # Run single task from config
  python cli.py trigger-task --task pools_screener --config config/data_collection_v2.yml
  
  # Run task directly with built-in defaults (no config needed!)
  python cli.py run app.tasks.data_collection.pools_screener
  python cli.py run app.tasks.data_collection.candles_downloader_task
  
  # Start API server with tasks
  python cli.py serve --config config/data_collection_v2.yml --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run tasks continuously
    run_parser = subparsers.add_parser('run-tasks', help='Run tasks continuously')
    run_parser.add_argument('--config', '-c', 
                           default='config/pools_screener_v2.yml',
                           help='Path to tasks configuration file')
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose logging')
    
    # Trigger single task
    trigger_parser = subparsers.add_parser('trigger-task', help='Run a single task once')
    trigger_parser.add_argument('--task', '-t', required=True,
                               help='Task name to trigger')
    trigger_parser.add_argument('--config', '-c',
                               default='config/pools_screener_v2.yml', 
                               help='Path to tasks configuration file')
    trigger_parser.add_argument('--timeout', type=int, default=300,
                               help='Task timeout in seconds')
    
    # Run task directly with built-in defaults
    direct_parser = subparsers.add_parser('run', help='Run a task directly with built-in defaults')
    direct_parser.add_argument('task_path', 
                              help='Task module path (e.g., app.tasks.data_collection.pools_screener)')
    direct_parser.add_argument('--timeout', type=int, default=600,
                              help='Task timeout in seconds')
    
    # Serve API with tasks
    serve_parser = subparsers.add_parser('serve', help='Start API server with background tasks')
    serve_parser.add_argument('--config', '-c',
                             default='config/pools_screener_v2.yml',
                             help='Path to tasks configuration file')
    serve_parser.add_argument('--port', '-p', type=int, default=8000,
                             help='API server port')
    serve_parser.add_argument('--host', default='0.0.0.0',
                             help='API server host')
    
    # List tasks
    list_parser = subparsers.add_parser('list-tasks', help='List available tasks')
    list_parser.add_argument('--config', '-c',
                            default='config/pools_screener_v2.yml',
                            help='Path to tasks configuration file')
    
    # Validate config  
    validate_parser = subparsers.add_parser('validate-config', help='Validate task configuration')
    validate_parser.add_argument('--config', '-c', required=True,
                                help='Path to configuration file to validate')
    
    return parser.parse_args()


async def run_tasks(config_path: str, verbose: bool = False):
    """Run tasks continuously."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting QuantsLab Task Runner v2.0")
    logger.info(f"Config: {config_path}")
    
    try:
        # Run tasks without API server (API disabled by default)
        runner = TaskRunner(config_path=config_path, enable_api=False)
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running tasks: {e}")
        sys.exit(1)


async def trigger_task(task_name: str, config_path: str, timeout: int):
    """Trigger a single task."""
    logger.info(f"Triggering task: {task_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Timeout: {timeout}s")
    
    try:
        runner = TaskRunner(config_path=config_path)
        
        # Setup storage and orchestrator
        storage_config = runner._get_storage_config()
        from core.tasks.storage import TimescaleDBTaskStorage
        storage = TimescaleDBTaskStorage(storage_config)
        
        from core.tasks.orchestrator import TaskOrchestrator
        max_concurrent = runner.config.get("max_concurrent_tasks", 10)
        runner.orchestrator = TaskOrchestrator(
            storage=storage,
            max_concurrent_tasks=max_concurrent,
            retry_failed_tasks=runner.config.get("retry_failed_tasks", True)
        )
        
        # Initialize tasks
        tasks = await runner._initialize_tasks()
        for task in tasks:
            runner.orchestrator.add_task(task)
        
        # Trigger specific task
        result = await runner.orchestrator.execute_task(
            task_name=task_name,
            force=True
        )
        
        if result:
            logger.info(f"Task {task_name} completed with status: {result.status}")
            if result.error_message:
                logger.error(f"Error: {result.error_message}")
                sys.exit(1)
        else:
            logger.error(f"Task {task_name} not found or could not be executed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error triggering task: {e}")
        sys.exit(1)


async def serve_api(config_path: str, host: str, port: int):
    """Start API server with background tasks."""
    logger.info(f"Starting QuantsLab API Server")
    logger.info(f"Config: {config_path}")
    logger.info(f"Server: http://{host}:{port}")
    
    try:
        # Create runner with API enabled and configure host/port
        runner = TaskRunner(config_path=config_path, enable_api=True)
        runner.api_host = host
        runner.api_port = port
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)


async def run_task_direct(task_path: str, timeout: int):
    """Run a task directly using its built-in main() function."""
    logger.info(f"Running task directly: {task_path}")
    logger.info(f"Timeout: {timeout}s")
    
    try:
        # Import the task module
        import importlib
        module = importlib.import_module(task_path)
        
        if not hasattr(module, 'main'):
            logger.error(f"Task module {task_path} does not have a main() function")
            sys.exit(1)
        
        # Execute with timeout
        await asyncio.wait_for(module.main(), timeout=timeout)
        logger.info(f"Task {task_path} completed successfully")
        
    except asyncio.TimeoutError:
        logger.error(f"Task {task_path} timed out after {timeout} seconds")
        sys.exit(1)
    except ImportError as e:
        logger.error(f"Failed to import task {task_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running task {task_path}: {e}")
        sys.exit(1)


def list_tasks(config_path: str):
    """List available tasks from configuration."""
    logger.info(f"Loading tasks from: {config_path}")
    
    try:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        
        runner = TaskRunner(config_path=config_path)
        tasks_config = runner.load_config()
        
        print("\nAvailable Tasks:")
        print("=" * 50)
        for task_name, task_config in tasks_config.get('tasks', {}).items():
            enabled = task_config.get('enabled', True)
            status = "✓ enabled" if enabled else "✗ disabled" 
            task_class = task_config.get('task_class', 'Unknown')
            schedule = task_config.get('schedule', {})
            schedule_info = f"({schedule.get('type', 'unknown')})"
            print(f"{task_name:30} {status:12} {task_class} {schedule_info}")
    
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        sys.exit(1)


def validate_config(config_path: str):
    """Validate task configuration file."""
    logger.info(f"Validating config: {config_path}")
    
    try:
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            sys.exit(1)
        
        runner = TaskRunner(config_path=config_path)
        tasks_config = runner.load_config()
        
        # Validate each task config
        errors = []
        for task_name, task_config in tasks_config.get('tasks', {}).items():
            try:
                TaskConfig(**task_config, name=task_name)
            except Exception as e:
                errors.append(f"Task {task_name}: {e}")
        
        if errors:
            logger.error("Validation errors found:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
        else:
            logger.info("✓ Config is valid")
            
    except Exception as e:
        logger.error(f"✗ Config validation failed: {e}")
        sys.exit(1)


async def main():
    args = parse_args()
    
    if args.command == 'run-tasks':
        await run_tasks(args.config, args.verbose)
    elif args.command == 'trigger-task':
        await trigger_task(args.task, args.config, args.timeout)
    elif args.command == 'run':
        await run_task_direct(args.task_path, args.timeout)
    elif args.command == 'serve':
        await serve_api(args.config, args.host, args.port)
    elif args.command == 'list-tasks':
        list_tasks(args.config)
    elif args.command == 'validate-config':
        validate_config(args.config)
    else:
        logger.error("No command specified. Use --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())