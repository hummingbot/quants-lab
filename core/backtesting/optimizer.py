import datetime
import os.path
import subprocess
import traceback
from abc import ABC, abstractmethod
from typing import List, Optional, Type
from dotenv import load_dotenv

import optuna
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.backtesting import BacktestingEngine
from core.services.timescale_client import TimescaleClient
from hummingbot.strategy_v2.backtesting.backtesting_engine_base import BacktestingEngineBase
from hummingbot.strategy_v2.controllers import ControllerConfigBase


load_dotenv()


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

    backtester = None

    def __init__(self, start_date: datetime.datetime, end_date: datetime.datetime,
                 backtester: Optional[BacktestingEngineBase] = None):
        """
        Initialize with common parameters for backtesting.

        Args:
            start_date (datetime.datetime): The start date of the backtesting period.
            end_date (datetime.datetime): The end date of the backtesting period.
        """
        self.start = int(start_date.timestamp())
        self.end = int(end_date.timestamp())
        self.backtester = backtester

    @abstractmethod
    async def generate_config(self, trial) -> BacktestingConfig:
        """
        Generate the configuration for a given trial.
        This method must be implemented by subclasses.

        Args:
            trial (optuna.Trial): The trial object containing hyperparameters to optimize.

        Returns:
            BacktestingConfig: An object containing the configuration, start time, and end time.
        """
        pass

    async def generate_custom_configs(self) -> List[BacktestingConfig]:
        """
        Generate custom configurations for optimization.
        This method must be implemented by subclasses.

        Returns:
            List[BacktestingConfig]: A list of objects containing the configuration, start time, and end time.
        """
        pass


class StrategyOptimizer:
    """
    Class for optimizing trading strategies using Optuna and a backtesting engine.
    """

    def __init__(self, load_cached_data: bool = False, resolution: str = "1m",
                 db_client: Optional[TimescaleClient] = None):
        """
        Initialize the optimizer with a backtesting engine and database configuration.

        Args:
            load_cached_data (bool): Whether to load cached backtesting data.
            resolution (str): The resolution or time frame of the data (e.g., '1h', '1d').
        """
        self._backtesting_engine = BacktestingEngine(load_cached_data=load_cached_data)
        self._db_client = db_client

        self.resolution = resolution
        self.optuna_postgres_user = os.getenv("POSTGRES_USER", "admin")
        self.optuna_postgres_password = os.getenv("POSTGRES_PASSWORD", "admin")
        self.optuna_postgres_host = os.getenv("OPTUNA_HOST", "localhost")
        self.optuna_postgres_db_name = "optimization_database"
        self.optuna_postgres_port = 5432 if self.optuna_postgres_host == "optuna-db" else 5433
        self._storage_name = f"postgresql://{self.optuna_postgres_user}:{self.optuna_postgres_password}@{self.optuna_postgres_host}:{self.optuna_postgres_port}/{self.optuna_postgres_db_name}"
        logging.info(f"Connecting to {self._storage_name}")
        self.dashboard_process = None

    def get_all_study_names(self):
        """
        Get all the study names available in the database.

        Returns:
            List[str]: A list of study names.
        """
        return optuna.get_all_study_names(self._storage_name)

    def get_study(self, study_name: str):
        """
        Get the study object for a given study name.

        Args:
            study_name (str): The name of the study.

        Returns:
            optuna.Study: The study object.
        """
        return optuna.load_study(study_name=study_name, storage=self._storage_name)

    def get_study_trials_df(self, study_name: str):
        """
        Get the trials data frame for a given study name.

        Args:
            study_name (str): The name of the study.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the trials data.
        """
        study = self.get_study(study_name)
        df = study.trials_dataframe()
        df.dropna(inplace=True)
        # Renaming the columns that start with 'user_attrs_'
        df.rename(columns={col: col.replace('user_attrs_', '') for col in df.columns if col.startswith('user_attrs_')},
                  inplace=True)
        df.rename(columns={col: col.replace('params_', '') for col in df.columns if col.startswith('params_')}, )
        return df

    def get_study_best_params(self, study_name: str):
        """
        Get the best parameters for a given study name.

        Args:
            study_name (str): The name of the study.

        Returns:
            Dict[str, Any]: A dictionary containing the best parameters.
        """
        study = self.get_study(study_name)
        return study.best_params

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
            sampler=optuna.samplers.TPESampler(),
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

    async def optimize_custom_configs(self, study_name: str, config_generator: Type[BaseStrategyConfigGenerator],
                                      load_if_exists: bool = True):
        """
        Run the optimization process asynchronously using custom configurations.

        Args:
            study_name (str): The name of the study.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            load_if_exists (bool): Whether to load an existing study if available.
        """
        study = self._create_study(study_name, load_if_exists=load_if_exists)
        await self._optimize_async_custom_configs(study, config_generator)

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

    async def _optimize_async_custom_configs(self, study: optuna.Study,
                                             config_generator: Type[BaseStrategyConfigGenerator]):
        """
        Asynchronously optimize using the provided study and configuration generator.

        Args:
            study (optuna.Study): The study to use for optimization.
            config_generator (Type[BaseStrategyConfigGenerator]): A configuration generator class instance.
            n_trials (int): Number of trials to run for optimization.
        """
        backtesting_configs = config_generator.generate_custom_configs()
        await self._db_client.connect()
        for bt_config in backtesting_configs:
            trial = study.ask()
            try:
                connector_name = bt_config.config.connector_name
                trading_pair = bt_config.config.trading_pair
                start = bt_config.start
                end = bt_config.end
                candles = await self._db_client.get_candles(connector_name,
                                                            trading_pair,
                                                            self.resolution, start, end)
                self._backtesting_engine._dt_bt.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{self.resolution}"] = candles.data
                self._backtesting_engine._mm_bt.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{self.resolution}"] = candles.data
                config_generator.backtester.backtesting_data_provider.candles_feeds[
                    f"{connector_name}_{trading_pair}_{self.resolution}"] = candles.data
                start = candles.data["timestamp"].min()
                end = candles.data["timestamp"].max()
                # Generate configuration using the config generator
                backtesting_result = await self._backtesting_engine.run_backtesting(
                    config=bt_config.config,
                    start=start,
                    end=end,
                    backtesting_resolution=self.resolution,
                    backtester=config_generator.backtester,
                )
                strategy_analysis = backtesting_result.results

                for key, value in strategy_analysis.items():
                    trial.set_user_attr(key, value)
                trial.set_user_attr("config", backtesting_result.controller_config.json())

                # Return the value you want to optimize
                value = strategy_analysis["net_pnl"]
            except Exception as e:
                print(f"An error occurred during optimization: {str(e)}")
                traceback.print_exc()
                value = float('-inf')  # Return a very low value to indicate failure

            # Report the result back to the study
            study.tell(trial, value)

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
            backtesting_config = await config_generator.generate_config(trial)

            # Await the backtesting result
            backtesting_result = await self._backtesting_engine.run_backtesting(
                config=backtesting_config.config,
                start=backtesting_config.start,
                end=backtesting_config.end,
                backtesting_resolution=self.resolution,
                backtester=config_generator.backtester,
            )
            strategy_analysis = backtesting_result.results

            for key, value in strategy_analysis.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("config", backtesting_result.controller_config.json())
            executors_df = backtesting_result.executors_df.copy()
            executors_df["close_type"] = executors_df["close_type"].apply(lambda x: x.name)
            executors_df["status"] = executors_df["status"].apply(lambda x: x.name)
            executors_df.drop(columns=["config", "custom_info"], inplace=True)
            trial.set_user_attr("executors", executors_df.to_json())

            # Return the value you want to optimize
            return strategy_analysis["sharpe_ratio"]
        except Exception as e:
            print(f"An error occurred during optimization: {str(e)}")
            traceback.print_exc()
            return float('-inf')  # Return a very low value to indicate failure

    def launch_optuna_dashboard(self):
        """
        Launch the Optuna dashboard for visualization.
        """
        self.dashboard_process = subprocess.Popen(["optuna-dashboard", self._storage_name])

    def kill_optuna_dashboard(self):
        """
        Kill the Optuna dashboard process.
        """
        if self.dashboard_process and self.dashboard_process.poll() is None:
            self.dashboard_process.terminate()  # Graceful termination
            self.dashboard_process.wait()  # Wait for process to terminate
            self.dashboard_process = None  # Reset process handle
        else:
            print("Dashboard is not running or already terminated.")
