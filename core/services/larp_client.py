from typing import List, Optional

from core.services.client_base import ClientBase


class LarpClient(ClientBase):
    def __init__(self, host: str = "localhost", port: int = 3000):
        super().__init__(host, port)

    # Solana endpoints
    async def get_balance(self, address: Optional[str] = None, symbols: List[str] = ["SOL"]):
        """Get token balances for the specified wallet address or the user's wallet if not provided."""
        endpoint = "solana/balance"
        params = {"symbols": symbols}
        if address:
            params["address"] = address
        return await self.get(endpoint, params=params)

    async def get_tokens(self):
        """List all tokens available in the Solana token list."""
        endpoint = "solana/tokens"
        return await self.get(endpoint)

    async def get_token_by_address(self, token_address: str, use_api: bool = False):
        """Retrieve info about a Solana token by address."""
        endpoint = f"solana/token/{token_address}"
        params = {"useApi": use_api}
        return await self.get(endpoint, params=params)

    async def get_token_by_symbol(self, symbol: str):
        """Retrieve info about a Solana token by symbol from the stored token list."""
        endpoint = f"solana/symbol/{symbol}"
        return await self.get(endpoint)

    # Jupiter endpoints
    async def get_jupiter_quote_swap(self, input_token_symbol: str, output_token_symbol: str, amount: float,
                                     slippage_pct: float = 1, only_direct_routes: bool = False,
                                     as_legacy_transaction: bool = False):
        """Get a swap quote for Jupiter."""
        endpoint = "jupiter/quote-swap"
        params = {
            "inputTokenSymbol": input_token_symbol,
            "outputTokenSymbol": output_token_symbol,
            "amount": amount,
            "slippagePct": slippage_pct,
            "onlyDirectRoutes": only_direct_routes,
            "asLegacyTransaction": as_legacy_transaction
        }
        return await self.get(endpoint, params=params)

    async def execute_jupiter_swap(self, input_token_symbol: str, output_token_symbol: str, amount: float,
                                   slippage_pct: float = 1):
        """Execute a swap on Jupiter."""
        endpoint = "jupiter/execute-swap"
        payload = {
            "inputTokenSymbol": input_token_symbol,
            "outputTokenSymbol": output_token_symbol,
            "amount": amount,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    # Orca endpoints
    async def get_orca_positions_owned(self, address: Optional[str] = None):
        """Retrieve a list of Orca positions owned by an address or the user's wallet."""
        endpoint = "orca/positions-owned"
        params = {"address": address} if address else None
        return await self.get(endpoint, params=params)

    async def get_orca_bundles_owned(self, address: Optional[str] = None):
        """Retrieve a list of Orca position bundles owned by an address or the user's wallet."""
        endpoint = "orca/bundles-owned"
        params = {"address": address} if address else None
        return await self.get(endpoint, params=params)

    async def get_orca_position(self, position_address: str):
        """Retrieve info about an Orca position."""
        endpoint = f"orca/position/{position_address}"
        return await self.get(endpoint)

    async def get_orca_quote_fees(self, position_address: str):
        """Get the fees quote for an Orca position."""
        endpoint = f"orca/quote-fees/{position_address}"
        return await self.get(endpoint)

    async def get_pool_info(self, pool_address: str):
        """Retrieve info about a pool."""
        endpoint = f"orca/pool/{pool_address}"
        return await self.get(endpoint)

    async def get_orca_quote_swap(self, input_token_symbol: str, output_token_symbol: str, amount: float,
                                  slippage_pct: float = 1, tick_spacing: int = 64):
        """Get a swap quote for Orca."""
        endpoint = "orca/quote-swap"
        params = {
            "inputTokenSymbol": input_token_symbol,
            "outputTokenSymbol": output_token_symbol,
            "amount": amount,
            "slippagePct": slippage_pct,
            "tickSpacing": tick_spacing
        }
        return await self.get(endpoint, params=params)

    async def execute_orca_swap(self, input_token_symbol: str, output_token_symbol: str, amount: float,
                                tick_spacing: int = 64, slippage_pct: float = 1):
        """Execute a swap on Orca."""
        endpoint = "orca/execute-swap"
        payload = {
            "inputTokenSymbol": input_token_symbol,
            "outputTokenSymbol": output_token_symbol,
            "amount": amount,
            "tickSpacing": tick_spacing,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def open_orca_position(self, base_symbol: str, quote_symbol: str, tick_spacing: int,
                                 lower_price: str, upper_price: str, quote_token_amount: float,
                                 slippage_pct: float = 1):
        """Open a new Orca position."""
        endpoint = "orca/open-position"
        payload = {
            "baseSymbol": base_symbol,
            "quoteSymbol": quote_symbol,
            "tickSpacing": tick_spacing,
            "lowerPrice": lower_price,
            "upperPrice": upper_price,
            "quoteTokenAmount": quote_token_amount,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def close_orca_position(self, position_address: str, slippage_pct: float = 1):
        """Close an Orca position."""
        endpoint = "orca/close-position"
        params = {"positionAddress": position_address, "slippagePct": slippage_pct}
        return await self.post(endpoint, params=params)

    async def add_liquidity_quote(self, position_address: str, quote_token_amount: float, slippage_pct: float = 1):
        """Get quote for adding liquidity to an Orca position."""
        endpoint = "orca/add-liquidity-quote"
        payload = {
            "positionAddress": position_address,
            "quoteTokenAmount": quote_token_amount,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def add_liquidity(self, position_address: str, quote_token_amount: float, slippage_pct: float = 1):
        """Add liquidity to an Orca position."""
        endpoint = "orca/add-liquidity"
        payload = {
            "positionAddress": position_address,
            "quoteTokenAmount": quote_token_amount,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def remove_liquidity(self, position_address: str, percentage_to_remove: float, slippage_pct: float = 1):
        """Remove liquidity from an Orca position."""
        endpoint = "orca/remove-liquidity"
        payload = {
            "positionAddress": position_address,
            "percentageToRemove": percentage_to_remove,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def collect_orca_fees(self, position_address: str):
        """Collect fees for an Orca position."""
        endpoint = f"orca/collect-fees/{position_address}"
        return await self.post(endpoint)

    async def get_orca_positions_in_bundle(self, position_bundle_address: str):
        """Retrieve info about all positions in an Orca position bundle."""
        endpoint = f"orca/positions-in-bundle/{position_bundle_address}"
        return await self.get(endpoint)

    async def collect_orca_fee_rewards(self, position_address: str):
        """Collect fees and rewards for an Orca position."""
        endpoint = f"orca/collect-fee-rewards/{position_address}"
        return await self.post(endpoint)

    async def create_orca_position_bundle(self):
        """Create a new Orca position bundle."""
        endpoint = "orca/create-position-bundle"
        return await self.post(endpoint, payload={})

    async def open_orca_positions_in_bundle(self, base_symbol: str, quote_symbol: str, tick_spacing: int,
                                            lower_price: str, upper_price: str, position_bundle_address: str,
                                            number_of_positions: int):
        """Open multiple new bundled Orca positions."""
        endpoint = "orca/open-positions-in-bundle"
        payload = {
            "baseSymbol": base_symbol,
            "quoteSymbol": quote_symbol,
            "tickSpacing": tick_spacing,
            "lowerPrice": lower_price,
            "upperPrice": upper_price,
            "positionBundleAddress": position_bundle_address,
            "numberOfPositions": number_of_positions
        }
        return await self.post(endpoint, payload=payload)

    async def add_liquidity_in_bundle(self, position_bundle_address: str, quote_token_amounts: List[float],
                                      slippage_pct: float = 1):
        """Add liquidity to multiple Orca positions in a bundle."""
        endpoint = "orca/add-liquidity-in-bundle"
        payload = {
            "positionBundleAddress": position_bundle_address,
            "quoteTokenAmounts": quote_token_amounts,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def remove_liquidity_in_bundle(self, position_bundle_address: str, percentages: List[float],
                                         slippage_pct: float = 1):
        """Remove liquidity from multiple Orca positions in a bundle."""
        endpoint = "orca/remove-liquidity-in-bundle"
        payload = {
            "positionBundleAddress": position_bundle_address,
            "percentages": percentages,
            "slippagePct": slippage_pct
        }
        return await self.post(endpoint, payload=payload)

    async def close_orca_positions_in_bundle(self, position_bundle_address: str):
        """Close all bundled Orca positions in a position bundle."""
        endpoint = "orca/close-positions-in-bundle"
        payload = {"positionBundleAddress": position_bundle_address}
        return await self.post(endpoint, payload=payload)

    async def delete_orca_position_bundle(self, position_bundle_address: str):
        """Delete an Orca position bundle."""
        endpoint = "orca/delete-position-bundle"
        params = {"positionBundleAddress": position_bundle_address}
        return await self.post(endpoint, params=params)
