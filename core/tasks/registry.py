"""
Task class registry for simplified task_class references.
"""

# Registry of simplified task class names to full paths
TASK_CLASS_REGISTRY = {
    "notebook": "app.tasks.notebook.enhanced_notebook_task.NotebookTask",
    "notebooks": "app.tasks.notebook.enhanced_notebook_task.NotebookTask",  # Alias
    
    # Add more simplified names here as needed
    # "data_collection": "app.tasks.data_collection.SomeTask",
    # "backtest": "app.tasks.backtesting.BacktestTask",
}


def resolve_task_class(task_class: str) -> str:
    """
    Resolve a task class name to its full module path.
    
    Args:
        task_class: Either a simplified name (e.g., "notebook") or full path
        
    Returns:
        Full module path for the task class
    """
    # If it's already a full path, return as-is
    if "." in task_class:
        return task_class
    
    # Look up in registry
    if task_class in TASK_CLASS_REGISTRY:
        return TASK_CLASS_REGISTRY[task_class]
    
    # If not found, return as-is (will probably fail later with helpful error)
    return task_class


def register_task_class(simple_name: str, full_path: str):
    """Register a new simplified task class name."""
    TASK_CLASS_REGISTRY[simple_name] = full_path


def list_registered_tasks() -> dict:
    """Get all registered simplified task class names."""
    return TASK_CLASS_REGISTRY.copy()