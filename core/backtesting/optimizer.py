import datetime
import os.path
import traceback
from abc import ABC, abstractmethod
from typing import Type

import optuna
from pydantic import BaseModel

from core.backtesting import BacktestingEngine
from hummingbot.strategy_v2.controllers import ControllerConfigBase


class BacktestingConfig(BaseModel):
    """
    A simple data structure to hold the backtesting configuration.
    """
    config: ControllerConfigBase
    start: int
    end: int


class BaseStrategyConfigGenerator(ABC):
    """
    Base class for generating strategy configurations for optimization.
    Subclasses should implement the method to provide specific strategy configurations.
    """

    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime):
        """
        Initialize with common parameters for backtesting.

        Args:
            start_date (datetime.datetime): The start date of the backtesting period.
            end_date (datetime.datetime): The end date of the backtesting period.
        """
        self.start = int(start_date.timestamp())
        self.end = int(end_date.timestamp())

    @abstractmethod
    def generate_config(self, trial) -> BacktestingConfig:
        """
        Generate the configuration for a given trial.
        This method must be implemented by subclasses.

        Args:
            trial (optuna.Trial): The trial object containing hyperparameters to optimize.

        Returns:
            BacktestingConfig: An object containing the configuration, start time, and end time.
        """
        pass


class StrategyOptimizer:
    """
    Class for optimizing trading strategies using Optuna and a backtesting engine.
    """

    def __init__(self, root_path: str = "", database_name: str = "optimization_database",
                 load_cached_data: bool = False, resolution: str = "1h"):
        """
        Initialize the optimizer with a backtesting engine and database configuration.

        Args:
            root_path (str): Root path for storing database files.
            database_name (str): Name of the SQLite database for storing optimization results.
            load_cached_data (bool): Whether to load cached backtesting data.
            resolution (str): The resolution or time frame of the data (e.g., '1h', '1d').
        """
        self._backtesting_engine = BacktestingEngine(load_cached_data=load_cached_data)
        self.resolution = resolution
        db_path = os.path.join(root_path, "data", "backtesting", f"{database_name}.db")
        self._storage_name = f"sqlite:///{db_path}"

    def _create_study(self, study_name: str, direction: str = "maximize", load_if_exists: bool = True) -> optuna.Study:
        """
        Create or load an Optuna study for optimization.

        Args:
            study_name (str): The name of the study.
            direction (str): Direction of optimization ("maximize" or "minimize").
            load_if_exists (bool): Whether to load an existing study if available.

        Returns:
            optuna.Study: The created or loaded study.
        """
        return optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=self._storage_name,
            load_if_exists=load_if_exists
        )

    async def optimize(self, study_name: str, config_generator: Type[BaseStrategyConfigGenerator], n_trials: int = 100,
                       load_if_exists: bool = True):
        """
        Run the optimization process asynchronously.

        Args:
            study_name (str): The name of the study.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            n_trials (int): Number of trials to run for optimization.
            load_if_exists (bool): Whether to load an existing study if available.
        """
        study = self._create_study(study_name, load_if_exists=load_if_exists)
        await self._optimize_async(study, config_generator, n_trials=n_trials)

    async def _optimize_async(self, study: optuna.Study, config_generator: Type[BaseStrategyConfigGenerator],
                              n_trials: int):
        """
        Asynchronously optimize using the provided study and configuration generator.

        Args:
            study (optuna.Study): The study to use for optimization.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            n_trials (int): Number of trials to run for optimization.
        """
        for _ in range(n_trials):
            trial = study.ask()

            try:
                # Run the async objective function and get the result
                value = await self._async_objective(trial, config_generator)

                # Report the result back to the study
                study.tell(trial, value)

            except Exception as e:
                print(f"Error in _optimize_async: {str(e)}")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

    async def _async_objective(self, trial: optuna.Trial, config_generator: Type[BaseStrategyConfigGenerator]) -> float:
        """
        The asynchronous objective function for a given trial.

        Args:
            trial (optuna.Trial): The trial object containing hyperparameters.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.

        Returns:
            float: The objective value to be optimized.
        """
        try:
            # Generate configuration using the config generator
            backtesting_config = config_generator.generate_config(trial)

            # Await the backtesting result
            backtesting_result = await self._backtesting_engine.run_backtesting(
                backtesting_config.config,
                backtesting_config.start,
                backtesting_config.end,
                self.resolution
            )
            strategy_analysis = backtesting_result.results

            for key, value in strategy_analysis.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("config", backtesting_result.controller_config.json())

            # Return the value you want to optimize
            return strategy_analysis["net_pnl"]
        except Exception as e:
            print(f"An error occurred during optimization: {str(e)}")
            traceback.print_exc()
            return float('-inf')  # Return a very low value to indicate failure
