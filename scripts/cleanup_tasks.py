#!/usr/bin/env python
"""
Utility script to clean up stale task states in MongoDB.
Use this when tasks appear to be stuck in a running state.
"""
import os
import sys
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def cleanup_stale_tasks():
    """Reset all tasks marked as running in MongoDB."""
    try:
        # Connect to MongoDB - use MONGO_URI from environment
        mongo_uri = os.getenv('MONGO_URI')
        mongo_db = os.getenv('MONGO_DATABASE', 'quants_lab')
        
        if not mongo_uri:
            print("✗ MONGO_URI environment variable not set")
            return 1
        
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        
        # Find all running tasks
        running_tasks = list(db['task_schedules'].find({'is_running': True}))
        
        if not running_tasks:
            print("✓ No tasks are currently marked as running")
            return 0
        
        print("Found tasks marked as running:")
        for task in running_tasks:
            print(f"  - {task.get('task_name')} (last updated: {task.get('updated_at')})")
        
        # Reset all running tasks
        result = db['task_schedules'].update_many(
            {'is_running': True},
            {
                '$set': {
                    'is_running': False,
                    'current_execution_id': None,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        print(f"\n✓ Reset {result.modified_count} task(s) to not running state")
        
        # Optional: Clean up old execution records (older than 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        result = db['task_executions'].delete_many({
            'started_at': {'$lt': week_ago}
        })
        if result.deleted_count > 0:
            print(f"✓ Cleaned up {result.deleted_count} old execution records")
        
        client.close()
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def list_task_states():
    """List current state of all tasks."""
    try:
        # Connect to MongoDB - use MONGO_URI from environment
        mongo_uri = os.getenv('MONGO_URI')
        mongo_db = os.getenv('MONGO_DATABASE', 'quants_lab')
        
        if not mongo_uri:
            print("✗ MONGO_URI environment variable not set")
            return 1
        
        client = MongoClient(mongo_uri)
        db = client[mongo_db]
        
        tasks = list(db['task_schedules'].find())
        
        if not tasks:
            print("No tasks found in database")
            return 0
        
        print("Current Task States:")
        print("-" * 60)
        for task in tasks:
            status = "🟢 RUNNING" if task.get('is_running') else "⚪ IDLE"
            print(f"{task.get('task_name'):30} {status}")
            if task.get('last_run'):
                print(f"  Last run: {task.get('last_run')}")
            if task.get('run_count'):
                print(f"  Total runs: {task.get('run_count')} (Success: {task.get('success_count', 0)}, Failed: {task.get('failure_count', 0)})")
        
        client.close()
        return 0
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


if __name__ == "__main__":
    from datetime import timedelta
    
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        sys.exit(list_task_states())
    else:
        print("MongoDB Task Cleanup Utility")
        print("=" * 40)
        sys.exit(cleanup_stale_tasks())