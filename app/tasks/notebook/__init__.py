"""
Notebook tasks module for executing Jupyter notebooks programmatically.

Provides enhanced notebook orchestration with support for:
- Single or multiple notebooks
- Sequential or parallel execution
- Parameter inheritance and smart configuration
- Comprehensive error handling
"""

from .notebook_task import NotebookTask

__all__ = ['NotebookTask']