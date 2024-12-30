import os
from solana.rpc.async_api import AsyncClient
from solana. import PublicKey
from solana.rpc.types import MemcmpOpts
from typing import List, Dict
import asyncio

class SolanaClient:
    def __init__(self):
        self.rpc_url = os.getenv('CHAINSTACK_SOLANA_RPC_URL')
        if not self.rpc_url:
            raise ValueError("Chainstack Solana RPC URL not found in environment variables")
        self.client = AsyncClient(self.rpc_url)

    async def get_token_info(self, token_address: str) -> Dict:
        """
        Get information about a token, including total supply and recent transactions.
        """
        try:
            token_pubkey = PublicKey(token_address)
            
            # Fetch token supply
            supply_info = await self.client.get_token_supply(token_pubkey)
            total_supply = float(supply_info["result"]["value"]["amount"]) / (10 ** supply_info["result"]["value"]["decimals"])

            # Fetch token accounts
            memcmp_opts = [MemcmpOpts(offset=0, bytes=token_address)]
            token_accounts = await self.client.get_program_accounts(
                PublicKey("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"), memcmp_opts=memcmp_opts
            )
            holder_count = len(token_accounts["result"])

            return {
                "total_supply": total_supply,
                "holder_count": holder_count,
                "recent_transactions": [],  # Add as needed
            }
        except Exception as e:
            raise Exception(f"Error fetching token info: {e}")

    async def get_account_info(self, account_address: str) -> Dict:
        """
        Get information about an account, including SOL and token balances.
        """
        try:
            account_pubkey = PublicKey(account_address)
            
            # Fetch SOL balance
            balance_response = await self.client.get_balance(account_pubkey)
            sol_balance = float(balance_response["result"]["value"]) / 1e9  # Convert lamports to SOL

            return {
                "sol_balance": sol_balance,
                "token_balances": [],  # Token parsing can be added here
                "recent_transactions": [],  # Add as needed
            }
        except Exception as e:
            raise Exception(f"Error fetching account info: {e}")

    async def close(self):
        await self.client.close()

# Example Usage
async def main():
    solana_client = SolanaClient()
    try:
        token_info = await solana_client.get_token_info("TokenAddressHere")
        print(token_info)
    finally:
        await solana_client.close()

asyncio.run(main())
