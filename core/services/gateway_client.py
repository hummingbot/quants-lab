from typing import List, Optional

from core.services.client_base import ClientBase

firstWalletAddress = "82SggYRE2Vo4jN4a2pk3aQ4SET4ctafZJGbowmCqyHx5"
class GatewayClient(ClientBase):
    def __init__(self, host: str = "localhost", port: int = 15888):
        super().__init__(host, port)

    async def get_connectors(self):
        """Returns a list of available DEX connectors and their supported blockchain networks."""
        endpoint = "connectors"
        return await self.get(endpoint)

    # Chain endpoints
    async def get_chain_status(
        self,
        chain: str = "solana", 
        network: str = "mainnet-beta",
        ):
        """Get the status of the gateway."""
        endpoint = f"{chain}/status"
        params = {
            "network": network,
        }
        return await self.get(endpoint, params=params)

    async def get_chain_tokens(
        self,
        chain: str = "solana", 
        network: str = "mainnet-beta",
        tokenSymbols: Optional[List[str]] = None,
        ):
        """List all tokens available in the Solana token list."""
        endpoint = f"{chain}/tokens"
        params = {"network": network}
        if tokenSymbols is not None:
            params["tokenSymbols"] = tokenSymbols
        return await self.get(endpoint, params=params)

    async def post_chain_balances(
        self,
        chain: str = "solana",  
        network: str = "mainnet-beta",
        address: Optional[str] = None,
        tokenSymbols: Optional[List[str]] = None,
        ):
        """Get token balances for the specified wallet address or the user's wallet if not provided."""
        endpoint = f"{chain}/balances"
        body = {
            "network": network,
            "tokenSymbols": tokenSymbols
        }
        if address:
            body["address"] = address
        return await self.post(endpoint, payload=body)
    
    async def post_chain_poll(
        self,
        txHash: str,
        chain: str = "solana",
        network: str = "mainnet-beta",
        ):
        """Poll for the status of a transaction on the specified blockchain network."""
        endpoint = f"{chain}/poll"
        body = {
            "network": network,
            "txHash": txHash
        }
        return await self.post(endpoint, payload=body)

