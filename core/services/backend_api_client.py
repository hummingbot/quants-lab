import json
import time
from typing import List, Optional

import aiohttp
import pandas as pd
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo

from core.services.client_base import ClientBase


class BackendAPIClient(ClientBase):
    """
    This class is a client to interact with the backend API. The Backend API is a REST API that provides endpoints to
    create new Hummingbot instances, start and stop them, add new script and controller config files, and get the status
    of the active bots.
    """
    _shared_instance = None

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "BackendAPIClient":
        if cls._shared_instance is None:
            cls._shared_instance = BackendAPIClient(*args, **kwargs)
        return cls._shared_instance

    def __init__(self, host: str = "localhost", port: int = 8000, username: str = "admin", password: str = "admin"):
        super().__init__(host, port)
        self.auth = aiohttp.BasicAuth(username, password)

    async def is_docker_running(self):
        """Check if Docker is running."""
        endpoint = "is-docker-running"
        return await self.get(endpoint, auth=self.auth)["is_docker_running"]

    async def pull_image(self, image_name: str):
        """Pull a Docker image."""
        endpoint = "pull-image"
        return await self.post(endpoint, payload={"image_name": image_name}, auth=self.auth)

    async def list_available_images(self, image_name: str):
        """List available images by name."""
        endpoint = f"available-images/{image_name}"
        return await self.get(endpoint, auth=self.auth)

    async def list_active_containers(self):
        """List all active containers."""
        endpoint = "active-containers"
        return await self.get(endpoint, auth=self.auth)

    async def list_exited_containers(self):
        """List all exited containers."""
        endpoint = "exited-containers"
        return await self.get(endpoint, auth=self.auth)

    async def clean_exited_containers(self):
        """Clean up exited containers."""
        endpoint = "clean-exited-containers"
        return await self.post(endpoint, payload=None, auth=self.auth)

    async def remove_container(self, container_name: str, archive_locally: bool = True, s3_bucket: str = None):
        """Remove a specific container."""
        endpoint = f"remove-container/{container_name}"
        params = {"archive_locally": archive_locally}
        if s3_bucket:
            params["s3_bucket"] = s3_bucket
        return await self.post(endpoint, payload=json.dumps(params), auth=self.auth)

    async def stop_container(self, container_name: str):
        """Stop a specific container."""
        endpoint = f"stop-container/{container_name}"
        return await self.post(endpoint, auth=self.auth)

    async def start_container(self, container_name: str):
        """Start a specific container."""
        endpoint = f"start-container/{container_name}"
        return await self.post(endpoint, auth=self.auth)

    async def create_hummingbot_instance(self, instance_config: dict):
        """Create a new Hummingbot instance."""
        endpoint = "create-hummingbot-instance"
        return await self.post(endpoint, payload=instance_config, auth=self.auth)

    async def start_bot(self, start_bot_config: dict):
        """Start a Hummingbot bot."""
        endpoint = "start-bot"
        return await self.post(endpoint, payload=start_bot_config, auth=self.auth)

    async def stop_bot(self, bot_name: str, skip_order_cancellation: bool = False, async_backend: bool = True):
        """Stop a Hummingbot bot."""
        endpoint = "stop-bot"
        return await self.post(endpoint,
                               payload={"bot_name": bot_name, "skip_order_cancellation": skip_order_cancellation,
                                        "async_backend": async_backend},
                               auth=self.auth)

    async def import_strategy(self, strategy_config: dict):
        """Import a trading strategy to a bot."""
        endpoint = "import-strategy"
        return await self.post(endpoint, payload=strategy_config, auth=self.auth)

    async def get_bot_status(self, bot_name: str):
        """Get the status of a bot."""
        endpoint = f"get-bot-status/{bot_name}"
        return await self.get(endpoint, auth=self.auth)

    async def get_bot_history(self, bot_name: str):
        """Get the historical data of a bot."""
        endpoint = f"get-bot-history/{bot_name}"
        return await self.get(endpoint, auth=self.auth)

    async def get_active_bots_status(self):
        """
        Retrieve the cached status of all active bots.
        Returns a JSON response with the status and data of active bots.
        """
        endpoint = "get-active-bots-status"
        return await self.get(endpoint, auth=self.auth)

    async def get_all_controllers_config(self):
        """Get all controller configurations."""
        endpoint = "all-controller-configs"
        return await self.get(endpoint, auth=self.auth)

    async def get_available_images(self, image_name: str = "hummingbot"):
        """Get available images."""
        endpoint = f"available-images/{image_name}"
        return await self.get(endpoint, auth=self.auth)["available_images"]

    async def add_script_config(self, script_config: dict):
        """Add a new script configuration."""
        endpoint = "add-script-config"
        return await self.post(endpoint, payload=script_config, auth=self.auth)

    async def add_controller_config(self, controller_config: dict):
        """Add a new controller configuration."""
        endpoint = "add-controller-config"
        config = {
            "name": controller_config["id"],
            "content": controller_config
        }
        return await self.post(endpoint, payload=config, auth=self.auth)

    async def delete_controller_config(self, controller_name: str):
        """Delete a controller configuration."""
        url = "delete-controller-config"
        return await self.post(url, params={"config_name": controller_name}, auth=self.auth)

    async def get_real_time_candles(self, connector: str, trading_pair: str, interval: str, max_records: int):
        """Get candles data."""
        endpoint = "real-time-candles"
        payload = {
            "connector": connector,
            "trading_pair": trading_pair,
            "interval": interval,
            "max_records": max_records
        }
        return await self.post(endpoint, payload=payload, auth=self.auth)

    async def get_historical_candles(self, connector: str, trading_pair: str, interval: str, start_time: int,
                                     end_time: int):
        """Get historical candles data."""
        endpoint = "historical-candles"
        payload = {
            "connector": connector,
            "trading_pair": trading_pair,
            "interval": interval,
            "start_time": start_time,
            "end_time": end_time
        }
        return await self.post(endpoint, payload=payload, auth=self.auth)

    async def run_backtesting(self, start_time: int, end_time: int, backtesting_resolution: str, trade_cost: float,
                              config: dict):
        """Run backtesting."""
        endpoint = "run-backtesting"
        payload = {
            "start_time": start_time,
            "end_time": end_time,
            "backtesting_resolution": backtesting_resolution,
            "trade_cost": trade_cost,
            "config": config
        }
        backtesting_results = await self.post(endpoint, payload=payload, auth=self.auth)
        if "error" in backtesting_results:
            raise Exception(backtesting_results["error"])
        if "processed_data" not in backtesting_results:
            data = None
        else:
            data = pd.DataFrame(backtesting_results["processed_data"])
        if "executors" not in backtesting_results:
            executors = []
        else:
            executors = [ExecutorInfo(**executor) for executor in backtesting_results["executors"]]
        return {
            "processed_data": data,
            "executors": executors,
            "results": backtesting_results["results"]
        }

    async def get_all_configs_from_bot(self, bot_name: str):
        """Get all configurations from a bot."""
        endpoint = f"all-controller-configs/bot/{bot_name}"
        return await self.get(endpoint, auth=self.auth)

    async def stop_controller_from_bot(self, bot_name: str, controller_id: str):
        """Stop a controller from a bot."""
        endpoint = f"update-controller-config/bot/{bot_name}/{controller_id}"
        config = {"manual_kill_switch": True}
        return await self.post(endpoint, payload=config, auth=self.auth)

    async def start_controller_from_bot(self, bot_name: str, controller_id: str):
        """Start a controller from a bot."""
        endpoint = f"update-controller-config/bot/{bot_name}/{controller_id}"
        config = {"manual_kill_switch": False}
        return await self.post(endpoint, payload=config, auth=self.auth)

    async def get_connector_config_map(self, connector_name: str):
        """Get connector configuration map."""
        endpoint = f"connector-config-map/{connector_name}"
        return await self.get(endpoint, auth=self.auth)

    async def get_all_connectors_config_map(self):
        """Get all connector configuration maps."""
        endpoint = "all-connectors-config-map"
        return await self.get(endpoint, auth=self.auth)

    async def add_account(self, account_name: str):
        """Add a new account."""
        endpoint = "add-account"
        return await self.post(endpoint, params={"account_name": account_name}, auth=self.auth)

    async def delete_account(self, account_name: str):
        """Delete an account."""
        endpoint = "delete-account"
        return await self.post(endpoint, params={"account_name": account_name}, auth=self.auth)

    async def delete_credential(self, account_name: str, connector_name: str):
        """Delete credentials."""
        endpoint = f"delete-credential/{account_name}/{connector_name}"
        return await self.post(endpoint, auth=self.auth)

    async def add_connector_keys(self, account_name: str, connector_name: str, connector_config: dict):
        """Add connector keys."""
        endpoint = f"add-connector-keys/{account_name}/{connector_name}"
        return await self.post(endpoint, payload=connector_config, auth=self.auth)

    async def get_accounts(self):
        """Get available credentials."""
        endpoint = "list-accounts"
        return await self.get(endpoint, auth=self.auth)

    async def get_credentials(self, account_name: str):
        """Get available credentials."""
        endpoint = f"list-credentials/{account_name}"
        return await self.get(endpoint, auth=self.auth)

    async def get_accounts_state(self):
        """Get all balances."""
        endpoint = "accounts-state"
        return await self.get(endpoint, auth=self.auth)

    async def get_account_state_history(self):
        """Get account state history."""
        endpoint = "account-state-history"
        return await self.get(endpoint, auth=self.auth)

    async def deploy_script_with_controllers(self,
                                             bot_name: str, controller_configs: List[str],
                                             script_name: str = "v2_with_controllers.py",
                                             image_name: str = "hummingbot/hummingbot:latest",
                                             credentials: str = "master_account",
                                             time_to_cash_out: Optional[int] = None,
                                             max_global_drawdown: Optional[float] = None,
                                             max_controller_drawdown: Optional[float] = None,
                                             ):
        start_time_str = time.strftime("%Y.%m.%d_%H.%M")
        bot_name = f"{bot_name}-{start_time_str}"
        script_config = {
            "name": bot_name,
            "content": {
                "markets": {},
                "candles_config": [],
                "controllers_config": controller_configs,
                "config_update_interval": 10,
                "script_file_name": script_name,
            }
        }
        if time_to_cash_out:
            script_config["content"]["time_to_cash_out"] = time_to_cash_out
        if max_global_drawdown:
            script_config["content"]["max_global_drawdown"] = max_global_drawdown
        if max_controller_drawdown:
            script_config["content"]["max_controller_drawdown"] = max_controller_drawdown

        await self.add_script_config(script_config)
        deploy_config = {
            "instance_name": bot_name,
            "script": script_name,
            "script_config": bot_name + ".yml",
            "image": image_name,
            "credentials_profile": credentials,
        }
        create_resp = await self.create_hummingbot_instance(deploy_config)
        return create_resp

    def list_databases(self):
        """Get databases list."""
        endpoint = "list-databases"
        return self.post(endpoint, auth=self.auth)

    def read_databases(self, db_paths: List[str]):
        """Read databases."""
        endpoint = "read-databases"
        return self.post(endpoint, payload=db_paths, auth=self.auth)
