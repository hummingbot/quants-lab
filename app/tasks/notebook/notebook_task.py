"""
Enhanced Notebook Task with simplified configuration and multi-notebook support.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import papermill as pm
from core.tasks import BaseTask, TaskContext, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class NotebookConfig:
    """Configuration for a single notebook."""
    
    def __init__(self, notebook_data: Union[str, Dict[str, Any]], global_params: Dict[str, Any] = None):
        if isinstance(notebook_data, str):
            # Simple string path
            self.path = notebook_data
            self.parameters = global_params or {}
        else:
            # Dictionary with path and parameters
            self.path = notebook_data.get("path", notebook_data.get("notebook_path", ""))
            local_params = notebook_data.get("parameters", {})
            # Merge global and local parameters (local overrides global)
            self.parameters = {**(global_params or {}), **local_params}
        
        if not self.path:
            raise ValueError("Notebook path is required")


class NotebookTask(BaseTask):
    """
    Enhanced notebook task with simplified configuration and multi-notebook support.
    
    Supports:
    - Single or multiple notebooks
    - Sequential or parallel execution
    - Parameter inheritance
    - Smart defaults
    - Minimal configuration
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Get task configuration
        task_config = self.config.config
        
        # Parse notebooks configuration (flexible input)
        notebooks_config = task_config.get("notebooks", [])
        if isinstance(notebooks_config, str):
            # Single notebook as string
            notebooks_config = [notebooks_config]
        elif not isinstance(notebooks_config, list):
            raise ValueError("'notebooks' must be a string or list")
        
        # Global parameters that apply to all notebooks
        global_parameters = task_config.get("parameters", {})
        
        # Parse notebook configurations
        self.notebooks = []
        for notebook_data in notebooks_config:
            self.notebooks.append(NotebookConfig(notebook_data, global_parameters))
        
        if not self.notebooks:
            raise ValueError("At least one notebook must be specified")
        
        # Execution configuration with smart defaults
        self.execution_mode = task_config.get("execution", "sequential").lower()
        if self.execution_mode not in ["sequential", "parallel"]:
            raise ValueError("execution must be 'sequential' or 'parallel'")
        
        self.max_parallel = task_config.get("max_parallel", 3)  # Reasonable default
        
        # Simple timeout parsing (supports formats like "30m", "1h", "120s")
        timeout_str = task_config.get("timeout", "30m")
        self.timeout_per_notebook = self._parse_timeout(timeout_str)
        
        # Smart defaults
        self.kernel_name = task_config.get("kernel", "python3")
        self.output_dir = Path(task_config.get("output_dir", "outputs/notebooks"))
        self.save_outputs = task_config.get("save_outputs", True)
        self.extract_results = task_config.get("extract_results", True)
        self.continue_on_error = task_config.get("continue_on_error", False)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Project root for resolving notebook paths
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.notebooks_base = self.project_root / "research_notebooks"
    
    def _parse_timeout(self, timeout_str: str) -> int:
        """Parse timeout string (e.g., '30m', '1h', '120s') to seconds."""
        if isinstance(timeout_str, (int, float)):
            return int(timeout_str)
        
        if not isinstance(timeout_str, str):
            return 1800  # Default 30 minutes
        
        timeout_str = timeout_str.strip().lower()
        
        # Extract number and unit
        if timeout_str.endswith('s'):
            return int(timeout_str[:-1])
        elif timeout_str.endswith('m'):
            return int(timeout_str[:-1]) * 60
        elif timeout_str.endswith('h'):
            return int(timeout_str[:-1]) * 3600
        else:
            # Assume seconds if no unit
            try:
                return int(timeout_str)
            except ValueError:
                return 1800  # Default 30 minutes
    
    def _resolve_notebook_path(self, notebook_path: str) -> Path:
        """Resolve notebook path relative to research_notebooks directory."""
        full_path = self.notebooks_base / notebook_path
        if not full_path.exists():
            raise FileNotFoundError(f"Notebook not found: {full_path}")
        return full_path
    
    async def setup(self, context: TaskContext) -> None:
        """Setup task before execution including validation of notebooks."""
        # Call parent setup to initialize database and notification services
        await super().setup(context)
        
        logger.info(f"Setting up enhanced notebook execution for {context.task_name}")
        logger.info(f"Notebooks: {len(self.notebooks)} notebook(s)")
        logger.info(f"Execution mode: {self.execution_mode}")
        logger.info(f"Max parallel: {self.max_parallel}")
        logger.info(f"Timeout per notebook: {self.timeout_per_notebook} seconds")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Validate that all notebooks exist and are readable
        for notebook_config in self.notebooks:
            notebook_path = self._resolve_notebook_path(notebook_config.path)
            
            if not notebook_path.is_file():
                raise RuntimeError(f"Not a file: {notebook_path}")
                
            if not notebook_path.suffix == '.ipynb':
                raise RuntimeError(f"Not a Jupyter notebook: {notebook_path}")
        
        for i, notebook_config in enumerate(self.notebooks):
            logger.info(f"  {i+1}. {notebook_config.path}")
            if notebook_config.parameters:
                logger.info(f"     Parameters: {json.dumps(notebook_config.parameters, indent=2)}")
    
    async def execute(self, context: TaskContext) -> Dict[str, Any]:
        """Execute notebooks according to configuration."""
        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting notebook execution: {len(self.notebooks)} notebook(s) in {self.execution_mode} mode")
        
        try:
            if self.execution_mode == "sequential":
                results = await self._execute_sequential()
            else:  # parallel
                results = await self._execute_parallel()
            
            # Aggregate results
            duration = datetime.now(timezone.utc) - start_time
            successful = sum(1 for r in results if r.get("status") == "completed")
            failed = len(results) - successful
            
            overall_result = {
                "status": "completed" if failed == 0 else ("partial" if successful > 0 else "failed"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_id": context.execution_id,
                "execution_mode": self.execution_mode,
                "total_notebooks": len(self.notebooks),
                "successful_notebooks": successful,
                "failed_notebooks": failed,
                "duration_seconds": duration.total_seconds(),
                "notebook_results": results
            }
            
            if failed > 0 and not self.continue_on_error:
                raise Exception(f"{failed} notebook(s) failed execution")
            
            logger.info(f"Notebook execution completed: {successful}/{len(self.notebooks)} successful")
            return overall_result
            
        except Exception as e:
            logger.error(f"Notebook execution failed: {e}")
            raise
    
    async def _execute_sequential(self) -> List[Dict[str, Any]]:
        """Execute notebooks sequentially."""
        results = []
        
        for i, notebook_config in enumerate(self.notebooks):
            logger.info(f"Executing notebook {i+1}/{len(self.notebooks)}: {notebook_config.path}")
            
            try:
                result = await self._execute_single_notebook(notebook_config, i+1)
                results.append(result)
                
                if result.get("status") != "completed" and not self.continue_on_error:
                    logger.error(f"Notebook {notebook_config.path} failed, stopping sequential execution")
                    break
                    
            except Exception as e:
                result = {
                    "notebook": notebook_config.path,
                    "status": "error",
                    "error": str(e),
                    "sequence_number": i+1
                }
                results.append(result)
                
                if not self.continue_on_error:
                    logger.error(f"Notebook {notebook_config.path} failed, stopping sequential execution")
                    break
        
        return results
    
    async def _execute_parallel(self) -> List[Dict[str, Any]]:
        """Execute notebooks in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def execute_with_semaphore(notebook_config: NotebookConfig, index: int):
            async with semaphore:
                return await self._execute_single_notebook(notebook_config, index+1)
        
        # Create tasks for parallel execution
        tasks = [
            execute_with_semaphore(notebook_config, i)
            for i, notebook_config in enumerate(self.notebooks)
        ]
        
        # Execute with error handling
        results = []
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                results.append({
                    "notebook": self.notebooks[i].path,
                    "status": "error",
                    "error": str(result),
                    "sequence_number": i+1
                })
            else:
                results.append(result)
        
        return results
    
    async def _execute_single_notebook(self, notebook_config: NotebookConfig, sequence_num: int) -> Dict[str, Any]:
        """Execute a single notebook."""
        start_time = datetime.now(timezone.utc)
        
        try:
            # Resolve paths
            input_path = self._resolve_notebook_path(notebook_config.path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_path.stem}_{sequence_num}_{timestamp}.ipynb"
            output_path = self.output_dir / output_filename
            
            # Execute notebook with Papermill (fixed for async context)
            await self._run_papermill_async(str(input_path), str(output_path), notebook_config.parameters)
            
            # Prepare result
            duration = datetime.now(timezone.utc) - start_time
            result = {
                "notebook": notebook_config.path,
                "status": "completed",
                "sequence_number": sequence_num,
                "output_path": str(output_path) if self.save_outputs else None,
                "duration_seconds": duration.total_seconds(),
                "parameters": notebook_config.parameters
            }
            
            # Extract results if configured
            if self.extract_results and output_path.exists():
                results_file = self.output_dir / f"results_{sequence_num}_{timestamp}.json"
                extracted = await self._extract_results(output_path, results_file)
                if extracted:
                    result["extracted_results"] = str(results_file)
            
            # Clean up output if not saving
            if not self.save_outputs and output_path.exists():
                output_path.unlink()
            
            return result
            
        except pm.PapermillExecutionError as e:
            # Handle notebook execution errors
            duration = datetime.now(timezone.utc) - start_time
            result = {
                "notebook": notebook_config.path,
                "status": "failed",
                "sequence_number": sequence_num,
                "duration_seconds": duration.total_seconds(),
                "error": str(e),
                "parameters": notebook_config.parameters
            }
            
            # Save partial outputs if available
            if 'output_path' in locals() and Path(output_path).exists() and self.save_outputs:
                result["output_path"] = str(output_path)
            
            return result
    
    async def _run_papermill_async(self, input_path: str, output_path: str, parameters: Dict[str, Any]):
        """Execute notebook with Papermill in a way that handles async context properly."""
        import concurrent.futures
        
        logger.info(f"Executing notebook with Papermill: {Path(input_path).name}")
        
        def run_papermill_sync():
            """Run papermill in a separate thread to avoid async context issues."""
            try:
                # Import here to avoid any module-level async issues
                import papermill as pm
                import nest_asyncio
                
                # Allow nested event loops for Jupyter notebooks with async code
                nest_asyncio.apply()
                
                pm.execute_notebook(
                    input_path=input_path,
                    output_path=output_path,
                    parameters=parameters,
                    kernel_name=self.kernel_name,
                    timeout=self.timeout_per_notebook,
                    progress_bar=False,
                    log_output=True
                )
                return True
            except Exception as e:
                logger.error(f"Papermill execution failed: {e}")
                raise e
        
        # Execute in a thread pool to avoid async context conflicts
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            await loop.run_in_executor(executor, run_papermill_sync)
        
        logger.info(f"Notebook executed successfully with Papermill: {Path(output_path).name}")
    
    async def _run_notebook_subprocess(self, input_path: str, output_path: str, parameters: Dict[str, Any]):
        """Execute notebook using subprocess to avoid async context issues."""
        import subprocess
        import sys
        import tempfile
        
        logger.info(f"Executing notebook via subprocess: {Path(input_path).name}")
        
        # Create parameterized notebook if parameters provided
        if parameters:
            temp_input = self._create_parameterized_notebook(input_path, parameters)
        else:
            temp_input = input_path
        
        try:
            # Use jupyter nbconvert for execution
            cmd = [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                f"--output={Path(output_path).name}",
                f"--output-dir={Path(output_path).parent}",
                f"--ExecutePreprocessor.timeout={self.timeout_per_notebook}",
                "--ExecutePreprocessor.kernel_name=" + self.kernel_name,
                temp_input
            ]
            
            logger.debug(f"Running command: {' '.join(cmd)}")
            
            # Run in subprocess with proper environment
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=None  # Inherit current environment
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_per_notebook + 60  # Extra buffer for subprocess overhead
            )
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else "Unknown error"
                logger.error(f"Notebook execution failed: {error_msg}")
                raise Exception(f"Notebook execution failed: {error_msg}")
            
            logger.info(f"Notebook executed successfully: {Path(output_path).name}")
            
        finally:
            # Clean up temporary parameterized notebook
            if parameters and temp_input != input_path:
                Path(temp_input).unlink(missing_ok=True)
    
    def _run_papermill(self, input_path: str, output_path: str, parameters: Dict[str, Any]):
        """Run papermill synchronously (for executor)."""
        try:
            pm.execute_notebook(
                input_path=input_path,
                output_path=output_path,
                parameters=parameters,
                kernel_name=self.kernel_name,
                timeout=self.timeout_per_notebook,
                progress_bar=False,
                log_output=True
            )
        except Exception as e:
            # Check if it's an async-related error
            if "cannot enter context" in str(e) or "RuntimeError" in str(e):
                logger.warning(f"Async context error detected, retrying with kernel restart")
                # Try to execute with fresh kernel environment
                import subprocess
                import sys
                
                # Use jupyter nbconvert as a fallback for async notebooks
                cmd = [
                    sys.executable, "-m", "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    f"--output={Path(output_path).name}",
                    f"--output-dir={Path(output_path).parent}",
                    "--ExecutePreprocessor.timeout={}".format(self.timeout_per_notebook),
                    input_path
                ]
                
                # Add parameters by creating a temporary notebook with injected parameters
                if parameters:
                    # Create temporary parameterized notebook
                    temp_nb = self._create_parameterized_notebook(input_path, parameters)
                    result = subprocess.run(cmd[:-1] + [temp_nb], 
                                          capture_output=True, text=True, 
                                          timeout=self.timeout_per_notebook + 30)
                    # Clean up temp file
                    Path(temp_nb).unlink(missing_ok=True)
                else:
                    result = subprocess.run(cmd, capture_output=True, text=True,
                                          timeout=self.timeout_per_notebook + 30)
                
                if result.returncode != 0:
                    raise Exception(f"Jupyter nbconvert failed: {result.stderr}")
            else:
                raise e
    
    def _create_parameterized_notebook(self, input_path: str, parameters: Dict[str, Any]) -> str:
        """Create a temporary notebook with parameters injected."""
        import nbformat
        import tempfile
        
        # Read the original notebook
        with open(input_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Find the parameters cell and update it
        for cell in nb.cells:
            if cell.cell_type == 'code' and 'parameters' in cell.get('metadata', {}).get('tags', []):
                # Inject parameters into the cell source
                param_lines = []
                for key, value in parameters.items():
                    if isinstance(value, str):
                        param_lines.append(f'{key} = "{value}"')
                    elif isinstance(value, list):
                        param_lines.append(f'{key} = {repr(value)}')
                    else:
                        param_lines.append(f'{key} = {value}')
                
                # Prepend parameters to existing cell content
                cell.source = '\n'.join(param_lines) + '\n\n' + cell.source
                break
        
        # Save to temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.ipynb')
        try:
            with open(temp_fd, 'w') as f:
                nbformat.write(nb, f)
        except:
            import os
            os.close(temp_fd)
            raise
        
        return temp_path
    
    async def _extract_results(self, notebook_path: Path, output_file: Path) -> bool:
        """Extract results from executed notebook."""
        try:
            import nbformat
            
            with open(notebook_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
            
            results = {}
            for i, cell in enumerate(nb.cells):
                if cell.cell_type == 'code' and cell.outputs:
                    for output in cell.outputs:
                        if hasattr(output, 'data'):
                            if 'application/json' in output.data:
                                results[f"cell_{i}"] = output.data['application/json']
                            elif 'text/plain' in output.data:
                                try:
                                    text_data = output.data['text/plain']
                                    if isinstance(text_data, str) and (
                                        text_data.startswith('{') or text_data.startswith('[')
                                    ):
                                        results[f"cell_{i}_text"] = json.loads(text_data)
                                except:
                                    pass
            
            if results:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Could not extract results: {e}")
            return False
    
    async def on_success(self, context: TaskContext, result: TaskResult) -> None:
        """Handle successful execution."""
        result_data = result.result_data or {}
        total = result_data.get("total_notebooks", 0)
        successful = result_data.get("successful_notebooks", 0)
        failed = result_data.get("failed_notebooks", 0)
        
        logger.info(f"✅ Enhanced NotebookTask completed in {result.duration_seconds:.2f}s")
        logger.info(f"   📊 Results: {successful}/{total} successful, {failed} failed")
        logger.info(f"   🏃 Mode: {result_data.get('execution_mode', 'unknown')}")
        
        notebook_results = result_data.get("notebook_results", [])
        for nb_result in notebook_results:
            status_emoji = "✅" if nb_result.get("status") == "completed" else "❌"
            logger.info(f"   {status_emoji} {nb_result.get('notebook', 'unknown')}")
    
    async def on_failure(self, context: TaskContext, result: TaskResult) -> None:
        """Handle failed execution."""
        result_data = result.result_data or {}
        logger.error(f"❌ Enhanced NotebookTask failed: {result.error_message}")
        
        notebook_results = result_data.get("notebook_results", [])
        for nb_result in notebook_results:
            if nb_result.get("status") != "completed":
                logger.error(f"   📓 {nb_result.get('notebook', 'unknown')}: {nb_result.get('error', 'Unknown error')}")
    
    async def cleanup(self, context: TaskContext, result: TaskResult) -> None:
        """Cleanup after task execution."""
        logger.info(f"Enhanced notebook task cleanup completed for {context.task_name}")


# Export alias for backwards compatibility