import asyncio
import argparse
from core.task_runner import TaskRunner

def parse_args():
    parser = argparse.ArgumentParser(description='Run tasks from configuration')
    parser.add_argument('--config', 
                       default='config/tasks.yml',
                       help='Path to tasks configuration file')
    return parser.parse_args()

async def main():
    args = parse_args()
    runner = TaskRunner(config_path=args.config)
    await runner.run()

if __name__ == "__main__":
    asyncio.run(main()) 