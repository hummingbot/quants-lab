import logging
import os
import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Set

from dotenv import load_dotenv

from core.task_base import BaseTask
from core.data_sources.clob import CLOBDataSource
from core.services.mongodb_client import MongoClient
from core.services.backend_api_client import BackendAPIClient
from tasks.deployment.models import ConfigCandidate

logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("hummingbot").setLevel(logging.ERROR)
load_dotenv()


class DeploymentBaseTask(BaseTask):
    rate_limit_config = {
        "okx_perpetual": {
            "batch_size": 3,
            "sleep_time": 10
        },
        "binance_perpetual": {
            "batch_size": 60,
            "sleep_time": 10,
        },
    }
    deploy_task_interval = 120.0
    control_task_interval = 3.0
    controller_stop_delay = 30.0

    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.backend_api_client = BackendAPIClient(self.config["backend_api_server"])
        self.mongo_client = MongoClient(self.config["mongo_uri"], database="quants_lab")
        self.root_path = "../../.."
        self.clob = CLOBDataSource()
        self.connector_name = self.config.get("connector_name", "binance_perpetual")
        self.connector_instance = None
        self.trading_rules = None
        self.trading_pairs = []
        self.config_candidates: List[ConfigCandidate] = []
        self.min_notionals_dict = {}
        self.last_prices: Dict[str, Any] = {}
        self.running = False
        self.active_bots: Dict[str, Any] = {}
        self.archived_configs: List[str] = []
        self.archived_bots: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()
        self.connector_instance = self.clob.get_connector(self.connector_name)
        await self.connector_instance._update_trading_rules()
        await self._update_exchange_info()
        self.running = True

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            tasks = [
                self._deploy_task(),
                self._control_task(),
            ]
            await asyncio.gather(*tasks)  # Corrected gathering of tasks
        except Exception as e:
            logging.exception(f"Error in Main Execution Task: {e}")
            raise

    async def _update_exchange_info(self):
        self.trading_rules = await self.clob.get_trading_rules(self.connector_name)
        return [trading_rule.trading_pair for trading_rule in self.trading_rules.data]

    async def _generate_config_candidates(self):
        """
        Asynchronously updates the list of configuration candidates by fetching all available configurations,
        filtering them based on the relevant trading pairs, and updating the internal state accordingly.
        """
        all_config_candidates = await self._fetch_controller_configs()
        ex_trading_pairs = await self._update_exchange_info()
        relevant_trading_pairs = self._extract_trading_pairs(all_config_candidates)
        filtered_trading_pairs = sorted(relevant_trading_pairs & set(ex_trading_pairs))
        self.trading_pairs = filtered_trading_pairs
        config_candidates = self._filter_configs_by_trading_pair(all_config_candidates, filtered_trading_pairs)
        return config_candidates

    async def _fetch_controller_configs(self) -> List[ConfigCandidate]:
        """
        Fetches a list of configuration candidates from an external source (e.g., a database or an API).

        Each configuration candidate typically consists of:
        - A dictionary containing the configuration settings.
        - Optional metadata for additional context.
        - A unique identifier (e.g., a MongoDB ObjectId) that can be useful for tracking or archiving purposes.

        This method must be implemented in a subclass or overridden to define the actual data retrieval logic.
        """
        raise NotImplementedError

    def _extract_trading_pairs(self, config_candidates: List[ConfigCandidate]) -> Set[str]:
        """
        Extracts the set of trading pairs associated with the given configuration candidates.

        Depending on the strategy, a configuration may correspond to a single trading pair or multiple pairs.
        This method should parse and return all relevant trading pairs from the provided configurations.

        This method must be implemented in a subclass or overridden to define the actual parsing logic.
        """
        raise NotImplementedError

    def _filter_configs_by_trading_pair(self, all_config_candidates: List[ConfigCandidate],
                                        trading_pairs: List[str]) -> List[ConfigCandidate]:
        """
        Filters the provided configuration candidates, keeping only those that match the specified trading pairs.

        The filtering criteria may depend on various factors such as exchange compatibility, strategy-specific
        requirements, or operational constraints.

        This method must be implemented in a subclass or overridden to define the actual filtering logic.
        """
        raise NotImplementedError

    async def _deploy_task(self):
        while self.running:
            try:
                bot_opportunity = await self._available_bot_slots()
                if bot_opportunity:
                    config_candidates = await self._generate_config_candidates()
                    if len(config_candidates) > 0:
                        await self._get_last_traded_prices()
                        selected_candidates = await self._filter_config_candidates(config_candidates)
                        if not selected_candidates:
                            logging.info(f"No config candidates found. Trying again in {self.deploy_task_interval} seconds")
                            await asyncio.sleep(self.deploy_task_interval)
                            continue
                        adjusted_candidates = self._adjust_config_candidates(selected_candidates)
                        logging.info(f"Config candidates found, preparing and launching bot...")
                        await self._prepare_and_launch_bots(adjusted_candidates)
            except Exception as e:
                logging.error(f"Error during deploy task: {e}")
            await asyncio.sleep(self.deploy_task_interval)

    async def _available_bot_slots(self):
        """Check if there are available bot slots based on backend response."""
        try:
            running_bots_data = await self.backend_api_client.get_active_bots_status()
            active_bots_resp = running_bots_data.get("data", {})
            active_bots = [bot_name for bot_name, _ in active_bots_resp.items() if bot_name in self.active_bots.keys()]
            n_active_bots = len(active_bots)
            max_bots = self.config["deploy_params"].get("max_bots", 1)
            return n_active_bots < max_bots

        except Exception as e:
            logging.exception(f"Error during _available_bot_slots execution: {e}")
            return False

    async def _get_last_traded_prices(self):
        try:
            self.last_prices = await self.connector_instance.get_last_traded_prices(self.trading_pairs)
            self._update_min_notional_size_dict()
        except Exception as e:
            logging.error(f"Error during get last traded prices: {e}")

    def _update_min_notional_size_dict(self):
        if self.connector_name in ["okx_perpetual"]:
            self.min_notionals_dict = {
                trading_rule.trading_pair: float(trading_rule.min_base_amount_increment) *
                self.last_prices[trading_rule.trading_pair] for trading_rule in self.trading_rules.data
                if self.last_prices.get(trading_rule.trading_pair) is not None}
        else:
            self.min_notionals_dict = {
                trading_rule.trading_pair: trading_rule.min_notional_size for trading_rule in self.trading_rules.data
            }

    async def _filter_config_candidates(self, config_candidates: List[ConfigCandidate]):
        filter_candidate_params = self.config.get("filter_candidate_params", {})
        if filter_candidate_params:
            selected_candidates = []
            for candidate in config_candidates:
                try:
                    config_already_used = candidate.id in self.archived_configs
                    if config_already_used and not self.config.get("recycle_configs"):
                        continue
                    meets_condition = await self._is_candidate_valid(candidate, filter_candidate_params)
                    if meets_condition:
                        selected_candidates.append(candidate)
                except Exception:
                    continue
            logging.info(f"Selected {len(selected_candidates)} out of {len(config_candidates)} candidates.")
            print(f"Selected {len(selected_candidates)} out of {len(config_candidates)} candidates.")
            return selected_candidates
        else:
            return config_candidates

    async def _is_candidate_valid(self, candidate: ConfigCandidate, filter_candidate_params: Dict[str, Any]) -> bool:
        """
        Determines whether a given configuration candidate is valid based on various filtering criteria.

        This method should be implemented in subclasses to apply specific validation rules based on
        the candidate's configuration and the provided filtering parameters.

        Args:
            candidate (ConfigCandidate): The candidate configuration to evaluate.
            filter_candidate_params (Dict[str, Any]): A dictionary containing the filtering parameters
                                                     used to determine validity.

        Returns:
            Boolean: True if the candidate meets the required conditions, False otherwise.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _adjust_config_candidates(self, config_candidates: List[ConfigCandidate]):
        """
        Adjusts configuration parameters for a list of configuration candidates.

        This method modifies each candidate's configuration based on predefined parameters,
        ensuring compliance with trading constraints such as leverage, minimum order amounts,
        and triple barrier configurations.

        Adjustments include:
        - Assigning a unique identifier to each candidate.
        - Updating total quote amount and order constraints.
        - Setting leverage and connector-related configurations.
        - Applying minimum spread and risk management parameters.

        Args:
            config_candidates (List[ConfigCandidate]): A list of configuration candidates
                                                       to be adjusted.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    async def _prepare_and_launch_bots(self, selected_candidates: List[ConfigCandidate]):
        script_name = self.config["deploy_params"].get("script_name", "v2_with_controllers.py")
        image_name = self.config["deploy_params"].get("image_name", "hummingbot/hummingbot:latest")
        credentials = self.config["deploy_params"].get("credentials", "master_account")
        time_to_cash_out = self.config["deploy_params"].get("time_to_cash_out")
        max_controller_configs = min(self.config["deploy_params"].get("max_controller_configs", 2), len(selected_candidates))

        # TODO: sort configs by multidimensional ghetman criteria
        # TODO: drop duplicate markets
        final_candidates = selected_candidates[:max_controller_configs]
        for candidate in final_candidates:
            await self.backend_api_client.add_controller_config(candidate.config)

        controller_configs = [candidate.config["id"] + ".yml" for candidate in final_candidates]
        year, iso_week = self.get_year_and_isoweek()
        bot_name = f"{self.config['connector_name']}-{year}-{iso_week}-{self.get_rounded_time()}"
        deploy_resp = await self.backend_api_client.deploy_script_with_controllers(bot_name=bot_name,
                                                                                   controller_configs=controller_configs,
                                                                                   script_name=script_name,
                                                                                   image_name=image_name,
                                                                                   credentials=credentials,
                                                                                   time_to_cash_out=time_to_cash_out)

        if deploy_resp["success"]:
            instance_name = self.extract_instance_name(deploy_resp)
            logging.info(f"Successfully deployed bot instance: {instance_name}")
            started_configs = [controller_id[:-4] for controller_id in controller_configs]
            self.active_bots[instance_name] = {
                "start_timestamp": time.time(),
                "controller_status": {controller_id: "running" for controller_id in started_configs}
            }
            self.archived_configs.extend([candidate.id for candidate in final_candidates])
        else:
            logging.error(f"There was an error trying to deploy: {deploy_resp['error']} - {deploy_resp['error']}")

    async def _control_task(self):
        while self.running:
            try:
                active_bots_data = await self.backend_api_client.get_active_bots_status()
                active_bots_resp = active_bots_data["data"] or {}
                active_bots = {bot_name: data for bot_name, data in active_bots_resp.items()
                               if bot_name in self.active_bots}
                if len(active_bots) == 0:
                    continue
                for bot_name, data in active_bots.items():
                    self._control_error_logs(data["error_logs"])
                    for controller_id, metrics in data["performance"].items():
                        if metrics["status"] == "running":
                            controller_info = metrics["performance"].copy()
                            controller_info["start_timestamp"] = self.active_bots[bot_name]["start_timestamp"]
                            controller_info["bot_name"] = bot_name
                            controller_info["controller_id"] = controller_id
                            self._control_pnl(controller_info)
            except Exception as e:
                logging.error(f"Error during control task: {e}")
            await asyncio.sleep(self.control_task_interval)

    def _control_error_logs(self, bot: Dict[str, Any]):
        pass  # TODO

    def _control_pnl(self, controller_info: Dict[str, Any]):
        global_pnl_pct = controller_info["global_pnl_pct"] / 100
        bot_name = controller_info["bot_name"]
        controller_id = controller_info["controller_id"]
        controller_max_drawdown = self.config["control_params"].get("controller_max_drawdown", 1.0)
        controller_max_pnl = self.config["control_params"].get("controller_max_pnl", 99.0)
        sl_condition = global_pnl_pct <= - controller_max_drawdown
        tp_condition = global_pnl_pct >= controller_max_pnl

        bot_duration = time.time() - controller_info["start_timestamp"]
        time_limit_condition = bot_duration >= self.config.get("global_time_limit", 10e10)

        min_early_stop_time = self.config["control_params"].get("min_early_stop_time", 10e10)
        max_early_stop_time = self.config["control_params"].get("max_early_stop_time", 10e11)
        early_tp_condition = early_sl_condition = False
        if min_early_stop_time <= bot_duration <= max_early_stop_time:
            partial_drawdown = self.config["control_params"].get("partial_drawdown", 1.0)
            partial_profit = self.config["control_params"].get("partial_profit", 1.0)
            early_tp_condition = global_pnl_pct >= partial_profit
            early_sl_condition = global_pnl_pct <= -partial_drawdown
        if tp_condition or sl_condition or time_limit_condition or early_tp_condition or early_sl_condition:
            asyncio.ensure_future(self._gracefully_stop_controller(bot_name, controller_id))

    async def _gracefully_stop_controller(self, bot_name: str, controller_id: str):
        await self.backend_api_client.stop_controller_from_bot(bot_name=bot_name, controller_id=controller_id)
        await asyncio.sleep(self.controller_stop_delay)
        self.active_bots[bot_name]["controller_status"][controller_id] = "stopped"
        running_controllers = [
            status for controller, status in self.active_bots[bot_name]["controller_status"].items()
            if status == "running"
        ]
        if not running_controllers:
            await self._gracefully_stop_bot_and_archive(bot_name)

    async def _gracefully_stop_bot_and_archive(self, bot_name: str):
        if bot_name in self.active_bots:
            await self.backend_api_client.stop_bot(bot_name=bot_name)
            await asyncio.sleep(self.controller_stop_delay)
            await self.backend_api_client.stop_container(bot_name)
            logging.info(f"Stopped container: {bot_name}")
            await asyncio.sleep(5.0)
            await self.backend_api_client.remove_container(bot_name, archive_locally=True)
            logging.info(f"Successfully archived bot!")
            self.archived_bots[bot_name] = self.active_bots[bot_name].copy()
            del self.active_bots[bot_name]

    @staticmethod
    def now():
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f UTC')

    @staticmethod
    def get_rounded_time():
        now = datetime.now()
        # Round to the nearest 10 minutes
        rounded_minute = (now.minute // 10) * 10
        rounded_time = now.replace(minute=rounded_minute, second=0, microsecond=0)

        formatted_time = f"{now.isoweekday()}-{rounded_time.strftime('%H%M')}"
        return formatted_time

    @staticmethod
    def get_year_and_isoweek():
        now = datetime.now()
        year, iso_week, _ = now.isocalendar()
        return year, iso_week

    @staticmethod
    def extract_instance_name(response: dict) -> str:
        """
        Extracts the instance full name from the response dictionary.

        Args:
            response (dict): A dictionary containing the instance creation message.

        Returns:
            str: The extracted instance name or an empty string if not found.
        """
        match = re.search(r'Instance (\S+) created successfully\.', response.get('message', ''))
        return match.group(1) if match else ""


async def main():
    connector_name = "binance_perpetual"
    mongo_uri = (
        f"mongodb://{os.getenv('MONGO_INITDB_ROOT_USERNAME', 'admin')}:"
        f"{os.getenv('MONGO_INITDB_ROOT_PASSWORD', 'admin')}@"
        f"{os.getenv('MONGO_HOST', 'localhost')}:"
        f"{os.getenv('MONGO_PORT', '27017')}/"
    )
    task_config = {
        "connector_name": connector_name,
        "mongo_uri": mongo_uri,
        "backend_api_server": os.getenv("BACKEND_API_SERVER", "localhost"),
        "min_config_timestamp": 1.5 * 24 * 60 * 60,
        "recycle_configs": False,
        "filter_candidate_params": {
            "max_base_step": 0.001,
            "max_quote_step": 0.001,
            "min_grid_range_ratio": 0.5,
            "max_grid_range_ratio": 2.0,
            "max_entry_price_distance": 0.5,
            "max_notional_size": 100.0
        },
        "config_adjustment_params": {
            "total_amount_quote": 1000.0,
            "min_spread_between_orders": 0.0004,
            "leverage": 50,
            "time_limit": 259200,
            "stop_loss": 0.1,
            "trailing_delta": 0.005,
            "take_profit": 0.0008,
            "activation_price": 0.03,
            "coerce_tp_to_step": True,
        },
        "deploy_params": {
            "max_bots": 1,
            "max_controller_configs": 2,
            "script_name": "v2_with_controllers.py",
            "image_name": "hummingbot/hummingbot:latest",
            "credentials": "master_account",
            "time_to_cash_out": 2 * 24 * 60 * 60,
        },
        "control_params": {
            "controller_max_drawdown": 0.005,
            "controller_max_pnl": 0.005,
            "global_time_limit": 24 * 60 * 60,
            "partial_drawdown": 0.1,
            "partial_profit": 0.1,
            "min_early_stop_time": 6 * 60 * 60,
            "max_early_stop_time": 24 * 60 * 60
        }
    }
    task = DeploymentBaseTask(name="deployment_task",
                              frequency=timedelta(minutes=20),
                              config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
