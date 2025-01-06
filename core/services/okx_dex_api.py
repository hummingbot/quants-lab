import base64
import hmac
import json
from datetime import datetime
import aiohttp
from typing import Optional, Dict, List, Union
import urllib.parse
from pydantic import BaseModel, Field
from decimal import Decimal
from base58 import b58decode, b58encode
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
import asyncio


class Chain(BaseModel):
    chain_id: str = Field(..., alias="chainId")
    chain_name: str = Field(..., alias="chainName")
    dex_token_approve_address: str = Field(..., alias="dexTokenApproveAddress")


class Token(BaseModel):
    decimals: str
    token_contract_address: str = Field(..., alias="tokenContractAddress")
    token_logo_url: str = Field(..., alias="tokenLogoUrl")
    token_name: str = Field(..., alias="tokenName")
    token_symbol: str = Field(..., alias="tokenSymbol")


class LiquiditySource(BaseModel):
    id: str
    logo: str
    name: str


class OKXResponse(BaseModel):
    code: str
    msg: str


class ChainsResponse(OKXResponse):
    data: List[Chain]


class TokensResponse(OKXResponse):
    data: List[Token]


class LiquiditySourcesResponse(OKXResponse):
    data: List[LiquiditySource]


class ApproveTransaction(BaseModel):
    data: str
    dex_contract_address: str = Field(..., alias="dexContractAddress")
    gas_limit: str = Field(..., alias="gasLimit")
    gas_price: str = Field(..., alias="gasPrice")


class ApproveResponse(OKXResponse):
    data: List[ApproveTransaction]


class TokenInfo(BaseModel):
    decimal: str
    is_honey_pot: bool = Field(..., alias="isHoneyPot")
    tax_rate: str = Field(..., alias="taxRate")
    token_contract_address: str = Field(..., alias="tokenContractAddress")
    token_symbol: str = Field(..., alias="tokenSymbol")
    token_unit_price: str = Field(..., alias="tokenUnitPrice")

    @property
    def price(self) -> Decimal:
        """Get token price in USD"""
        return Decimal(self.token_unit_price)


class DexProtocol(BaseModel):
    dex_name: str = Field(..., alias="dexName")
    percent: str


class SubRouter(BaseModel):
    dex_protocol: List[DexProtocol] = Field(..., alias="dexProtocol")
    from_token: TokenInfo = Field(..., alias="fromToken")
    to_token: TokenInfo = Field(..., alias="toToken")


class DexRouter(BaseModel):
    router: str
    router_percent: str = Field(..., alias="routerPercent")
    sub_router_list: List[SubRouter] = Field(..., alias="subRouterList")


class QuoteCompare(BaseModel):
    amount_out: str = Field(..., alias="amountOut")
    dex_logo: str = Field(..., alias="dexLogo") 
    dex_name: str = Field(..., alias="dexName")
    trade_fee: str = Field(..., alias="tradeFee")

    def get_output_amount(self) -> Decimal:
        """Get output amount as Decimal with proper decimals"""
        return Decimal(self.amount_out)

    def get_price(self, input_amount: Decimal) -> Decimal:
        """Calculate price (output per input) for this venue"""
        output_amount = self.get_output_amount()
        if input_amount == 0:
            return Decimal('0')
        return input_amount / output_amount  # Price in terms of input/output


class RouterResult(BaseModel):
    chain_id: str = Field(..., alias="chainId")
    dex_router_list: List[DexRouter] = Field(..., alias="dexRouterList")
    estimate_gas_fee: str = Field(..., alias="estimateGasFee")
    from_token: TokenInfo = Field(..., alias="fromToken")
    from_token_amount: str = Field(..., alias="fromTokenAmount")
    price_impact_pct: Optional[str] = Field(None, alias="priceImpactPercentage")
    quote_compare_list: List[QuoteCompare] = Field(..., alias="quoteCompareList")
    to_token: TokenInfo = Field(..., alias="toToken")
    to_token_amount: str = Field(..., alias="toTokenAmount")
    trade_fee: str = Field(..., alias="tradeFee")
    origin_to_token_amount: Optional[str] = Field(None, alias="originToTokenAmount")

    @property
    def from_amount_decimal(self) -> Decimal:
        """Get from amount as Decimal"""
        return Decimal(self.from_token_amount) / Decimal(10 ** int(self.from_token.decimal))

    @property
    def to_amount_decimal(self) -> Decimal:
        """Get to amount as Decimal"""
        return Decimal(self.to_token_amount) / Decimal(10 ** int(self.to_token.decimal))

    @property
    def execution_price(self) -> Decimal:
        """Calculate execution price (input per output)"""
        if self.to_amount_decimal == 0:
            return Decimal('0')
        return self.from_amount_decimal / self.to_amount_decimal

    @property
    def value_in_usd(self) -> Decimal:
        """Calculate the USD value of the trade"""
        return self.from_amount_decimal * self.from_token.price

    @property
    def price_impact(self) -> Optional[Decimal]:
        """Get price impact as a decimal (if available)"""
        return Decimal(self.price_impact_pct) if self.price_impact_pct else None

    @property
    def best_venue(self) -> QuoteCompare:
        """Get the venue offering the best price"""
        return min(self.quote_compare_list, 
                  key=lambda x: x.get_price(self.from_amount_decimal))

    def get_venue_prices(self) -> Dict[str, Decimal]:
        """Get a mapping of venue names to their prices (input/output)"""
        return {
            quote.dex_name: quote.get_price(self.from_amount_decimal)
            for quote in self.quote_compare_list
        }

    def get_price_comparison(self) -> str:
        """Get a formatted string comparing prices across venues"""
        prices = self.get_venue_prices()
        if not prices:
            return "No prices available"
            
        best_price = min(prices.values())  # Lower price is better when measuring input/output
        
        comparison = f"Best price: {best_price:.8f} {self.from_token.token_symbol}/{self.to_token.token_symbol}\n"
        comparison += "Prices by venue:\n"
        for venue, price in sorted(prices.items(), key=lambda x: x[1]):
            diff = ((price / best_price) - 1) * 100
            comparison += f"  {venue}: {price:.8f} ({diff:+.2f}%)\n"
        return comparison

class SwapTransaction(BaseModel):
    data: str
    from_address: str = Field(..., alias="from")
    gas: str
    gas_price: str = Field(..., alias="gasPrice")
    max_priority_fee_per_gas: Optional[str] = Field(None, alias="maxPriorityFeePerGas")
    to: str
    value: str
    min_receive_amount: Optional[str] = Field(None, alias="minReceiveAmount")


class SwapInfo(BaseModel):
    router_result: RouterResult = Field(..., alias="routerResult")
    tx: SwapTransaction


class SwapResponse(OKXResponse):
    data: List[SwapInfo]


class QuoteResponse(OKXResponse):
    data: List[RouterResult]

    @property
    def result(self) -> RouterResult:
        """Get the first (and usually only) result"""
        return self.data[0]


class BroadcastTransactionData(BaseModel):
    order_id: str = Field(..., alias="orderId")

class BroadcastTransactionResponse(OKXResponse):
    data: List[BroadcastTransactionData]

class TransactionOrder(BaseModel):
    chain_index: str = Field(..., alias="chainIndex")
    address: str
    account_id: Optional[str] = Field(None, alias="accountId")
    order_id: str = Field(..., alias="orderId")
    tx_status: str = Field(..., alias="txStatus")
    tx_hash: str = Field(..., alias="txHash")

class TransactionOrdersResponse(OKXResponse):
    data: List[TransactionOrder]

class OKXDexAPI:
    """
    OKX DEX API client implementing REST authentication
    """
    
    # API Endpoints
    SUPPORTED_CHAINS = "api/v5/dex/aggregator/supported/chain"
    ALL_TOKENS = "api/v5/dex/aggregator/all-tokens" 
    GET_LIQUIDITY = "api/v5/dex/aggregator/get-liquidity"
    APPROVE_TRANSACTION = "api/v5/dex/aggregator/approve-transaction"
    GET_QUOTE = "api/v5/dex/aggregator/quote"
    SWAP = "api/v5/dex/aggregator/swap"
    SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"  # Can be made configurable
    BROADCAST_TRANSACTION = "api/v5/wallet/pre-transaction/broadcast-transaction"
    GET_TRANSACTION_ORDERS = "api/v5/wallet/post-transaction/orders"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        access_project: Optional[str] = None,
        solana_private_key: Optional[str] = None,
        base_url: str = "https://www.okx.com",
        solana_rpc_url: Optional[str] = None,
    ):
        """
        Initialize the OKX DEX API client
        
        Args:
            api_key: The API key from OKX developer portal
            secret_key: The secret key from OKX developer portal  
            passphrase: The passphrase specified when creating API key
            solana_private_key: Optional Solana private key
            base_url: Base API URL, defaults to production
        """
        self.api_key = api_key
        self.secret_key = secret_key.encode()
        self.passphrase = passphrase
        self.access_project = access_project
        self.base_url = base_url.rstrip('/')
        self.tokens: List[Token] = []
        self.liquidity_sources: List[LiquiditySource] = []
        self.chains: List[Chain] = []
        self.solana_client = AsyncClient(solana_rpc_url or self.SOLANA_RPC_URL)
        self.solana_private_key = solana_private_key


    def _generate_signature(
        self, 
        timestamp: str,
        method: str,
        request_path: str,
        body: str = ""
    ) -> str:
        """
        Generate the signature required for API authentication
        
        Args:
            timestamp: ISO format timestamp
            method: HTTP method (GET/POST)
            request_path: API endpoint path including query params
            body: Request body for POST requests
            
        Returns:
            Base64 encoded signature
        """
        message = timestamp + method + request_path + body
        mac = hmac.new(
            self.secret_key,
            message.encode(),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()

    def _get_timestamp(self) -> str:
        """Get ISO format timestamp"""
        return datetime.utcnow().isoformat()[:-3] + 'Z'

    def _get_headers(
        self,
        method: str,
        request_path: str,
        body: str = ""
    ) -> Dict[str, str]:
        """
        Generate headers required for API authentication
        
        Args:
            method: HTTP method (GET/POST)
            request_path: API endpoint path including query params
            body: Request body for POST requests
            
        Returns:
            Dict of required headers
        """
        timestamp = self._get_timestamp()
        
        headers = {
            'Content-Type': 'application/json',
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': self._generate_signature(timestamp, method, request_path, body),
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase
        }
        if self.access_project:
            headers['OK-ACCESS-PROJECT'] = self.access_project
        return headers

    async def get(
        self,
        path: str,
        params: Optional[Dict] = None
    ) -> Dict:
        """
        Make a GET request to the API
        
        Args:
            path: API endpoint path
            params: Optional query parameters
            
        Returns:
            API response as dictionary
        """
        # Build full request path with query params
        request_path = f"/{path}"
        if params:
            query_string = urllib.parse.urlencode(params)
            request_path = f"{request_path}?{query_string}"

        url = f"{self.base_url}{request_path}"
        headers = self._get_headers("GET", request_path)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Error: {response.status} - {await response.text()}")
                        return {
                            "status": response.status,
                            "error": await response.text()
                        }
            except Exception as e:
                print(f"Request failed: {str(e)}")
                return {"error": str(e)}

    async def post(
        self,
        path: str,
        data: Dict
    ) -> Dict:
        """
        Make a POST request to the API
        
        Args:
            path: API endpoint path
            data: Request body data
            
        Returns:
            API response as dictionary
        """
        request_path = f"/{path}"
        url = f"{self.base_url}{request_path}"
        
        # Convert body to JSON string
        body = json.dumps(data)
        headers = self._get_headers("POST", request_path, body)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Error: {response.status} - {await response.text()}")
                        return {
                            "status": response.status,
                            "error": await response.text()
                        }
            except Exception as e:
                print(f"Request failed: {str(e)}")
                return {"error": str(e)}

    async def get_supported_chains(self) -> ChainsResponse:
        """
        Get chains available that support single-chain transactions
        
        Returns:
            List of supported chains and their details
        """
        response = await self.get(self.SUPPORTED_CHAINS)
        chains = ChainsResponse(**response)
        self.chains = chains.data
        return chains

    async def get_tokens(self, chain_id: str) -> TokensResponse:
        """
        Get tokens available for swap in the OKX aggregation protocol
        
        Args:
            chain_id: Chain ID (e.g., "1" for Ethereum)
            
        Returns:
            List of supported tokens and their details
        """
        params = {"chainId": chain_id}
        response = await self.get(self.ALL_TOKENS, params)
        tokens = TokensResponse(**response)
        self.tokens = tokens.data
        return tokens

    async def get_liquidity_sources(self, chain_id: str) -> LiquiditySourcesResponse:
        """
        Get liquidity sources available for swap in the OKX aggregation protocol
        
        Args:
            chain_id: Chain ID (e.g., "1" for Ethereum)
            
        Returns:
            List of liquidity sources and their details
        """
        params = {"chainId": chain_id}
        response = await self.get(self.GET_LIQUIDITY, params)
        liquidity_sources = LiquiditySourcesResponse(**response)
        self.liquidity_sources = liquidity_sources.data
        return liquidity_sources

    async def approve_transaction(
        self,
        chain_id: str,
        token_contract_address: str,
        approve_amount: str
    ) -> ApproveResponse:
        """
        Generate approval transaction data for EVM chains
        
        Args:
            chain_id: Chain ID (e.g., "1" for Ethereum)
            token_contract_address: Token contract address to approve
            approve_amount: Amount to approve in minimal divisible units
            
        Returns:
            Transaction data for approval
        """
        params = {
            "chainId": chain_id,
            "tokenContractAddress": token_contract_address,
            "approveAmount": approve_amount
        }
        response = await self.get(self.APPROVE_TRANSACTION, params)
        return ApproveResponse(**response)

    async def get_quote(
        self,
        chain_id: str,
        from_token_address: str,
        to_token_address: str,
        amount: str,
        fee_percent: Optional[str] = None
    ) -> QuoteResponse:
        """
        Get quote for token swap
        
        Args:
            chain_id: Chain ID (e.g., "1" for Ethereum)
            from_token_address: Address of token to swap from
            to_token_address: Address of token to swap to
            amount: Amount to swap, we will convert to minimal divisible units
            
        Returns:
            Quote information including price impact and routing
        """
        from_token = next((token for token in self.tokens if token.token_contract_address == from_token_address), None)
        to_token = next((token for token in self.tokens if token.token_contract_address == to_token_address), None)
        if not from_token or not to_token:
            raise ValueError("Token not found in supported tokens")
        
        amount_in_minimal_units = int(Decimal(amount) * Decimal(10 ** int(from_token.decimals)))
        params = {
            "chainId": chain_id,
            "fromTokenAddress": from_token_address,
            "toTokenAddress": to_token_address,
            "amount": amount_in_minimal_units
        }
        if fee_percent:
            params["feePercent"] = fee_percent
        response = await self.get(self.GET_QUOTE, params)
        return QuoteResponse(**response)

    async def swap(
        self,
        chain_id: str,
        from_token_address: str,
        to_token_address: str,
        amount: str,
        slippage: str,
        user_wallet_address: str
    ) -> SwapResponse:
        """
        Generate swap transaction data
        
        Args:
            chain_id: Chain ID (e.g., "1" for Ethereum)
            from_token_address: Address of token to swap from
            to_token_address: Address of token to swap to
            amount: Amount to swap, we will convert to   minimal divisible units
            slippage: Maximum acceptable slippage (e.g., "0.05" for 5%)
            user_wallet_address: User's wallet address
            
        Returns:
            Swap transaction data and routing information
        """
        from_token = next((token for token in self.tokens if token.token_contract_address == from_token_address), None)
        to_token = next((token for token in self.tokens if token.token_contract_address == to_token_address), None)
        if not from_token or not to_token:
            raise ValueError("Token not found in supported tokens")
        
        amount_in_minimal_units = int(Decimal(amount) * Decimal(10 ** int(from_token.decimals)))
        params = {
            "chainId": chain_id,
            "fromTokenAddress": from_token_address,
            "toTokenAddress": to_token_address,
            "amount": amount_in_minimal_units,
            "slippage": slippage,
            "userWalletAddress": user_wallet_address
        }
        response = await self.get(self.SWAP, params)
        return SwapResponse(**response)

    async def execute_solana_swap(
        self,
        from_token_address: str,
        to_token_address: str,
        amount: str,
        slippage: str,
        wallet_address: str,
        private_key: Optional[str] = None,
        poll_for_confirmation: bool = False,
        poll_sleep_seconds: Optional[float] = None
    ) -> str:
        """Execute a Solana swap transaction following Solders documentation"""
        # Verify private key
        private_key = private_key or self.solana_private_key
        if not private_key:
            raise ValueError("Private key required for Solana transactions")

        # 1. Get swap transaction data
        swap_response = await self.swap(
            chain_id="501",
            from_token_address=from_token_address,
            to_token_address=to_token_address,
            amount=amount,
            slippage=slippage,
            user_wallet_address=wallet_address,
        )

        # 2. Get latest blockhash
        recent_blockhash = await self.solana_client.get_latest_blockhash()
        
        # 3. Create keypair from private key bytes
        fee_payer = Keypair.from_bytes(b58decode(private_key))

        try:
            # 4. Decode transaction bytes and get min receive amount
            tx_bytes = b58decode(swap_response.data[0].tx.data)
            original_tx = VersionedTransaction.from_bytes(tx_bytes)
            
            # 5. Create new message with updated blockhash and min receive amount
            # Keep original account ordering and metadata
            new_message = MessageV0(
                header=original_tx.message.header,
                account_keys=original_tx.message.account_keys,
                recent_blockhash=recent_blockhash.value.blockhash,
                instructions=original_tx.message.instructions,
                address_table_lookups=original_tx.message.address_table_lookups,
            )
            
            # 6. Create and sign transaction
            tx = VersionedTransaction(new_message, [fee_payer])
            
            # 7. Send and confirm with more robust options
            opts = TxOpts(
                skip_preflight=True,  # Skip preflight to avoid local validation
                preflight_commitment="confirmed",
                max_retries=10
            )
            result = await self.solana_client.send_transaction(
                tx,
                opts=opts
            )
            if poll_for_confirmation:
                await self.poll_for_confirmation(result.value, poll_sleep_seconds)
            
            return result.value

        except Exception as e:
            raise ValueError(f"Transaction failed: {e}")
        
    async def poll_for_confirmation(self, tx_sig: str, sleep_seconds: float = 0.5):
        await self.solana_client.confirm_transaction(
            tx_sig=tx_sig,
            commitment="confirmed",
            sleep_seconds=sleep_seconds
        )

    async def broadcast_transaction(
        self,
        signed_tx: str,
        chain_index: str,
        address: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> str:
        """
        Broadcast a signed transaction through OKX
        
        Args:
            signed_tx: The signed transaction data
            chain_index: Chain identifier (e.g. "501" for Solana)
            address: Optional wallet address
            account_id: Optional account ID
            
        Returns:
            Transaction order ID
        """
        data = {
            "signedTx": signed_tx,
            "chainIndex": chain_index,
        }
        if address:
            data["address"] = address
        if account_id:
            data["accountId"] = account_id

        response = await self.post(self.BROADCAST_TRANSACTION, data)
        broadcast_response = BroadcastTransactionResponse(**response)
        return broadcast_response.data[0].order_id

    async def get_transaction_orders(
        self,
        address: Optional[str] = None,
        account_id: Optional[str] = None,
        chain_index: Optional[str] = None,
        tx_status: Optional[str] = None,
        order_id: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: Optional[str] = None,
    ) -> TransactionOrdersResponse:
        """
        Get list of transaction orders from OKX
        
        Args:
            address: Filter by wallet address
            account_id: Filter by account ID
            chain_index: Filter by chain index
            tx_status: Filter by status (1: Pending, 2: Success, 3: Failed)
            order_id: Filter by order ID
            cursor: Pagination cursor
            limit: Number of records (default 20, max 100)
            
        Returns:
            List of transaction orders
        """
        params = {}
        if address:
            params["address"] = address
        if account_id:
            params["accountId"] = account_id
        if chain_index:
            params["chainIndex"] = chain_index
        if tx_status:
            params["txStatus"] = tx_status
        if order_id:
            params["orderId"] = order_id
        if cursor:
            params["cursor"] = cursor
        if limit:
            params["limit"] = limit

        response = await self.get(self.GET_TRANSACTION_ORDERS, params)
        return TransactionOrdersResponse(**response)

    async def execute_solana_swap_via_okx(
        self,
        from_token_address: str,
        to_token_address: str,
        amount: str,
        slippage: str,
        wallet_address: str,
        private_key: Optional[str] = None
    ) -> str:
        """
        Execute a Solana swap transaction through OKX broadcast API
        Returns the order ID for tracking
        """
        # Get swap transaction data
        raw_swap = await self.swap(
            chain_id="501", 
            from_token_address=from_token_address,
            to_token_address=to_token_address,
            amount=amount,
            slippage=slippage,
            user_wallet_address=wallet_address
        )

        try:
            # Get latest blockhash
            recent_blockhash = await self.solana_client.get_latest_blockhash()

            # Create keypair from private key bytes
            private_key = private_key or self.solana_private_key
            fee_payer = Keypair.from_bytes(b58decode(private_key))

            # Decode transaction bytes
            tx_bytes = b58decode(raw_swap.data[0].tx.data)
            original_tx = VersionedTransaction.from_bytes(tx_bytes)

            # Create new message with updated blockhash
            new_message = MessageV0(
                header=original_tx.message.header,
                account_keys=original_tx.message.account_keys,
                recent_blockhash=recent_blockhash.value.blockhash,
                instructions=original_tx.message.instructions,
                address_table_lookups=original_tx.message.address_table_lookups,
            )

            # Create and sign new transaction
            tx = VersionedTransaction(new_message, [fee_payer])
            
            # Convert transaction to base58 string for OKX API
            signed_tx_str = b58encode(bytes(tx)).decode('utf-8')

            # Broadcast the signed transaction
            order_id = await self.broadcast_transaction(
                signed_tx=signed_tx_str,
                chain_index="501",
                address=wallet_address
            )
            return order_id

        except Exception as e:
            raise ValueError(f"Transaction failed: {e}")
