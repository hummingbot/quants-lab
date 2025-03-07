import logging
import math
import os
import asyncio
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

from dotenv import load_dotenv
from pydantic import BaseModel

from core.task_base import BaseTask
from core.data_sources.clob import CLOBDataSource
from core.services.mongodb_client import MongoClient
from core.services.backend_api_client import BackendAPIClient
import research_notebooks.statarb_v2.stat_arb_performance_utils as utils


logging.getLogger("asyncio").setLevel(logging.CRITICAL)
logging.getLogger("hummingbot").setLevel(logging.ERROR)
load_dotenv()


class ConfigCandidate(BaseModel):
    config: Dict[str, Any]
    extra_info: Dict[str, Any]
    id: str

    @classmethod
    def from_mongo(cls, data):
        """Convert MongoDB document to Pydantic model."""
        data["id"] = str(data["_id"])  # Convert ObjectId to string
        return cls(**data)


class ConfigProposal(BaseModel):
    config: Dict[str, Any]
    extra_info: Dict[str, Any]


class DeploymentTask(BaseTask):
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
    last_prices_update_interval = 10.0
    config_candidates_update_interval = 3600.0
    deploy_task_interval = 10.0
    control_task_interval = 3.0
    controller_stop_delay = 15.0

    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name=name, frequency=frequency, config=config)
        self.backend_api_client = BackendAPIClient(self.config["backend_api_server"])
        self.mongo_client = MongoClient(self.config["mongo_uri"], database="quants_lab")
        self.root_path = "../../.."
        self.clob = CLOBDataSource()
        self.connector_instance = None
        self.trading_rules = None
        self.ex_trading_pairs = []
        self.trading_pairs = []
        self.config_candidates: List[ConfigCandidate] = []
        self.min_notionals_dict = {}
        self.last_prices: Dict[str, Any] = {}
        self.running = False
        self.active_bots: Dict[str, Any] = {}
        self.archived_configs: List[str] = []
        self._config_candidates_available = asyncio.Event()
        self._bot_opportunity = asyncio.Event()

    async def initialize(self):
        """Initialize connections and resources."""
        await self.mongo_client.connect()
        self.connector_instance = self.clob.get_connector(self.config["connector_name"])
        await self.connector_instance._update_trading_rules()
        await self._update_exchange_info()
        self.running = True

    async def execute(self):
        """Main task execution logic."""
        try:
            await self.initialize()
            tasks = [
                self._update_config_candidates_task(),
                self._get_last_traded_prices(),
                self._deploy_task(),
                self._control_task(),
            ]
            await asyncio.gather(*tasks)  # Corrected gathering of tasks
        except Exception as e:
            logging.exception(f"Error in Main Execution Task: {e}")
            raise

    async def _update_config_candidates_task(self):
        while self.running:
            try:
                await self._bot_opportunity.wait()
                await self._update_exchange_info()
                await self._update_config_candidates()
            except Exception as e:
                logging.error(f"Error during update config candidates task: {e}")
            await asyncio.sleep(self.config_candidates_update_interval)

    async def _update_exchange_info(self):
        self.trading_rules = await self.clob.get_trading_rules(self.config["connector_name"])
        self.ex_trading_pairs = [trading_rule.trading_pair for trading_rule in self.trading_rules.data]

    async def _update_config_candidates(self):
        min_timestamp = self.config.get("min_config_timestamp", 24 * 60 * 60)
        controller_configs_data = await self.mongo_client.get_controller_config_data(min_timestamp=min_timestamp)
        config_candidates = [ConfigCandidate.from_mongo(config_data) for config_data in controller_configs_data]

        trading_pairs = {
                            candidate.config["base_trading_pair"] for candidate in config_candidates
                        } | {
                            candidate.config["quote_trading_pair"] for candidate in config_candidates
                        }

        self.trading_pairs = sorted(trading_pairs & set(self.ex_trading_pairs))

        self.config_candidates = [
            candidate for candidate in config_candidates
            if candidate.config["base_trading_pair"] in self.trading_pairs
            or candidate.config["quote_trading_pair"] in self.trading_pairs
        ]

        self._config_candidates_available.set() if self.config_candidates else self._config_candidates_available.clear()

    async def _get_last_traded_prices(self):
        while self.running:
            try:
                await self._config_candidates_available.wait()
                self.last_prices = await self.connector_instance.get_last_traded_prices(self.trading_pairs)
                self._update_min_notional_size_dict()
            except Exception as e:
                logging.error(f"Error during get last traded prices: {e}")
            await asyncio.sleep(self.last_prices_update_interval)

    def _update_min_notional_size_dict(self):
        if self.config["connector_name"] in ["okx_perpetual"]:
            self.min_notionals_dict = {
                trading_rule.trading_pair: float(trading_rule.min_base_amount_increment) *
                self.last_prices[trading_rule.trading_pair] for trading_rule in self.trading_rules.data
                if self.last_prices.get(trading_rule.trading_pair) is not None}
        else:
            self.min_notionals_dict = {
                trading_rule.trading_pair: trading_rule.min_notional_size for trading_rule in self.trading_rules.data
            }

    async def _deploy_task(self):
        while self.running:
            try:
                bot_opportunity = await self._available_bot_slots()
                if bot_opportunity:
                    await self._config_candidates_available.wait()
                    selected_candidates = await self._filter_config_candidates()
                    self._adjust_config_candidates(selected_candidates)
                    if not selected_candidates:
                        logging.info(f"No config candidates found. Trying again in {self.deploy_task_interval} seconds")
                        await asyncio.sleep(self.deploy_task_interval)
                        continue
                    logging.info(f"Config candidates found, preparing and launching bot...")
                    await self._prepare_and_launch_bots(selected_candidates)
            except Exception as e:
                logging.error(f"Error during deploy task: {e}")
            await asyncio.sleep(self.deploy_task_interval)

    async def _available_bot_slots(self):
        """Check if there are available bot slots based on backend response."""
        try:
            running_bots_data = await self.backend_api_client.get_active_bots_status()
            active_bots = running_bots_data.get("data", [])
            n_active_bots = len(active_bots)
            max_bots = self.config["deploy_params"].get("max_bots", 1)
            if n_active_bots < max_bots:
                self._bot_opportunity.set()
                return True
            else:
                self._bot_opportunity.clear()
                return False

        except Exception as e:
            logging.exception("Error during _available_bot_slots execution")
            return False

    async def _filter_config_candidates(self):
        selected_candidates = []
        error_configs = 0
        for candidate in self.config_candidates:
            try:
                meets_condition = await utils.apply_filters(
                    connector_instance=self.connector_instance,
                    config=candidate.config,
                    base_entry_price=self.last_prices.get(candidate.config["base_trading_pair"]),
                    quote_entry_price=self.last_prices.get(candidate.config["quote_trading_pair"]),
                    **self.config["filter_params"]
                )
                if meets_condition and candidate.id not in self.archived_configs:
                    selected_candidates.append(candidate)
            except Exception as e:
                error_configs += 1
        return selected_candidates

    def _adjust_config_candidates(self, config_candidates: List[ConfigCandidate]):
        params = self.config["config_adjustment_params"]
        for candidate in config_candidates:
            year, iso_week = self.get_year_and_isoweek()
            time_formatted = f"{year}||isoweek{iso_week}"
            connector_name = candidate.config["connector_name"]
            base_trading_pair = candidate.config["base_trading_pair"]
            quote_trading_pair = candidate.config["quote_trading_pair"]
            controller_id = f"{connector_name.replace('_', '-')}||{base_trading_pair}||{quote_trading_pair}||{time_formatted}"
            tag = self.get_rounded_time()
            candidate.config["id"] = f"{controller_id}_{tag}"

            candidate.config["total_amount_quote"] = params["total_amount_quote"]
            candidate.config["coerce_tp_to_step"] = params["coerce_tp_to_step"]

            base_min_order_amount_quote = float(self.min_notionals_dict[base_trading_pair]) * 1.5
            quote_min_order_amount_quote = float(self.min_notionals_dict[quote_trading_pair]) * 1.5
            min_order_amount_quote = max(base_min_order_amount_quote, quote_min_order_amount_quote)
            candidate.config["grid_config_base"]["min_order_amount_quote"] = min_order_amount_quote
            candidate.config["grid_config_quote"]["min_order_amount_quote"] = min_order_amount_quote

            candidate.config["leverage"] = params["leverage"]
            candidate.config["connector_name"] = self.config["connector_name"]
            candidate.config["min_spread_between_orders"] = params["min_spread_between_orders"]
            candidate.config["triple_barrier_config"] = {
                'stop_loss': params["stop_loss"],
                'take_profit': params["take_profit"],
                'time_limit': params["time_limit"],
                'trailing_stop': {
                    'activation_price': params["activation_price"],
                    'trailing_delta': params["trailing_delta"]
                }
            }

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
                active_bots = active_bots_data["data"] or []
                if len(active_bots) == 0:
                    continue
                for bot_name, stats in active_bots.items():
                    self._control_error_logs(stats["error_logs"])
                    for controller_id, metrics in stats["performance"].items():
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
            partial_drawdown = self.config["control_params"].get("partial_drawdown", -1.0)
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
        await self.backend_api_client.stop_bot(bot_name=bot_name)
        await asyncio.sleep(self.controller_stop_delay)
        await self.backend_api_client.stop_container(bot_name)
        logging.info(f"Stopped container: {bot_name}")
        await asyncio.sleep(5.0)
        await self.backend_api_client.remove_container(bot_name, archive_locally=True)
        logging.info(f"Successfully archived bot!")

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
        "min_config_timestamp": time.time() - 1.5 * 24 * 60 * 60,
        "filter_params": {
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
    task = DeploymentTask(name="deployment_task",
                          frequency=timedelta(minutes=20),
                          config=task_config)
    await task.execute()


if __name__ == "__main__":
    asyncio.run(main())
