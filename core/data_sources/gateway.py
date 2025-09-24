import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Gateway HTTP Client from hummingbot library
from hummingbot.core.gateway.gateway_http_client import GatewayHttpClient
from hummingbot.client.config.client_config_map import GatewayConfigMap

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GatewayDataSource:
    """Gateway data source for blockchain interactions via Hummingbot Gateway HTTP Client"""

    def __init__(self, gateway_url: Optional[str] = None, network: str = 'mainnet-beta'):
        logger.info("Initializing GatewayDataSource")

        # Configuration
        self.gateway_url = gateway_url or os.getenv('GATEWAY_URL', 'http://localhost:15888')
        self.network = network

        # Initialize Gateway Client
        self.gateway_config = GatewayConfigMap()
        self.gateway_config.gateway_api_host = self.gateway_url.replace('http://', '').replace('https://', '').split(':')[0]
        self.gateway_config.gateway_api_port = str(self.gateway_url.split(':')[-1])
        self.gateway_config.gateway_use_ssl = False

        # Get the Gateway client instance
        self.client = GatewayHttpClient.get_instance(self.gateway_config)

        # Cache for token information to avoid redundant API calls
        self.token_cache: Dict[str, Optional[Dict]] = {}

        logger.info(f"Gateway configured: {self.gateway_url} (Network: {network})")

    async def ping_gateway(self) -> bool:
        """Test Gateway connection"""
        try:
            return await self.client.ping_gateway()
        except Exception as e:
            logger.error(f"Gateway ping failed: {e}")
            return False

    async def get_gateway_status(self) -> Dict[str, Any]:
        """Get Gateway status information"""
        try:
            response = await self.client.api_request(
                method="get",
                path_url="status"
            )
            return response
        except Exception as e:
            logger.error(f"Error fetching Gateway status: {e}")
            return {}

    async def fetch_token_info(self, token_address: str, network: Optional[str] = None) -> Optional[Dict]:
        """Fetch token information from Gateway with caching"""
        network = network or self.network

        # Check cache first
        cache_key = f"{token_address}_{network}"
        if cache_key in self.token_cache:
            return self.token_cache[cache_key]

        try:
            response = await self.client.api_request(
                method="get",
                path_url=f"tokens/{token_address}",
                params={
                    'chain': 'solana',
                    'network': network
                }
            )
            token_info = response.get('token', {})
            # Cache the result
            self.token_cache[cache_key] = token_info
            return token_info
        except Exception as e:
            logger.error(f"Error fetching token info for {token_address}: {e}")
            # Cache the failure so we don't retry
            self.token_cache[cache_key] = None
            return None

    async def fetch_token_list(self, chain: str = 'solana', network: Optional[str] = None) -> List[Dict]:
        """Fetch list of available tokens"""
        network = network or self.network

        try:
            response = await self.client.api_request(
                method="get",
                path_url="tokens",
                params={
                    'chain': chain,
                    'network': network
                }
            )
            return response.get('tokens', [])
        except Exception as e:
            logger.error(f"Error fetching token list: {e}")
            return []

    async def get_token_price(self, token_address: str, network: Optional[str] = None) -> Optional[float]:
        """Get current token price"""
        network = network or self.network

        try:
            response = await self.client.api_request(
                method="get",
                path_url=f"price",
                params={
                    'chain': 'solana',
                    'network': network,
                    'connector': 'uniswap',  # or appropriate DEX
                    'tokenSymbol': token_address
                }
            )
            return float(response.get('price', 0))
        except Exception as e:
            logger.error(f"Error fetching token price for {token_address}: {e}")
            return None

    @staticmethod
    def format_number(value: float, decimals: int = 2, is_currency: bool = False) -> str:
        """Format numbers for display with appropriate units"""
        if pd.isna(value) or value is None:
            return "N/A"

        abs_value = abs(value)
        sign = "-" if value < 0 else ""

        if is_currency:
            if abs_value >= 1_000_000_000:
                return f"{sign}${abs_value/1_000_000_000:.{decimals}f}B"
            elif abs_value >= 1_000_000:
                return f"{sign}${abs_value/1_000_000:.{decimals}f}M"
            elif abs_value >= 1_000:
                return f"{sign}${abs_value/1_000:.{decimals}f}K"
            else:
                return f"{sign}${abs_value:.{decimals}f}"
        else:
            if abs_value >= 1_000_000_000:
                return f"{sign}{abs_value/1_000_000_000:.{decimals}f}B"
            elif abs_value >= 1_000_000:
                return f"{sign}{abs_value/1_000_000:.{decimals}f}M"
            elif abs_value >= 1_000:
                return f"{sign}{abs_value/1_000:.{decimals}f}K"
            else:
                return f"{sign}{abs_value:.{decimals}f}"

    @staticmethod
    def format_price(price: float) -> str:
        """Format price with appropriate decimal places based on magnitude"""
        if pd.isna(price) or price is None or price == 0:
            return "$0.00"

        if price >= 100:
            return f"${price:.2f}"
        elif price >= 10:
            return f"${price:.3f}"
        elif price >= 1:
            return f"${price:.4f}"
        elif price >= 0.1:
            return f"${price:.5f}"
        elif price >= 0.01:
            return f"${price:.6f}"
        elif price >= 0.001:
            return f"${price:.7f}"
        elif price >= 0.0001:
            return f"${price:.8f}"
        elif price > 0:
            # For very small prices, use scientific notation if needed
            if price < 0.00000001:
                return f"${price:.2e}"
            else:
                return f"${price:.10f}"
        else:
            return "$0.00"

    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format percentage values"""
        if pd.isna(value) or value is None:
            return "N/A"
        return f"{value:.{decimals}f}%"

    @staticmethod
    def format_timestamp(timestamp: datetime, format_str: str = '%Y-%m-%d %H:%M UTC') -> str:
        """Format timestamp for display"""
        if timestamp is None:
            return "N/A"
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.strftime(format_str)

    def create_basic_chart(self,
                          data: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          title: str = "Chart",
                          x_label: str = "X",
                          y_label: str = "Y",
                          chart_type: str = "bar",
                          color_col: Optional[str] = None,
                          hover_data: Optional[List[str]] = None) -> go.Figure:
        """Create a basic chart with common formatting"""

        fig = go.Figure()

        # Prepare hover template
        hover_template = f"{x_label}: %{{x}}<br>{y_label}: %{{y}}"
        custom_data = []

        if hover_data:
            for col in hover_data:
                if col in data.columns:
                    hover_template += f"<br>{col}: %{{customdata[{len(custom_data)}]}}"
                    custom_data.append(data[col])

        hover_template += "<extra></extra>"

        # Create trace based on chart type
        if chart_type == "bar":
            colors = data[color_col] if color_col and color_col in data.columns else None

            fig.add_trace(
                go.Bar(
                    x=data[x_col],
                    y=data[y_col],
                    name=y_label,
                    marker=dict(color=colors, opacity=0.7) if colors is not None else None,
                    hovertemplate=hover_template,
                    customdata=list(zip(*custom_data)) if custom_data else None
                )
            )
        elif chart_type == "scatter":
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='markers',
                    name=y_label,
                    hovertemplate=hover_template,
                    customdata=list(zip(*custom_data)) if custom_data else None
                )
            )
        elif chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=data[x_col],
                    y=data[y_col],
                    mode='lines',
                    name=y_label,
                    hovertemplate=hover_template,
                    customdata=list(zip(*custom_data)) if custom_data else None
                )
            )

        # Update layout
        fig.update_xaxes(title_text=x_label, showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
        fig.update_yaxes(title_text=y_label, showgrid=True, gridwidth=1, gridcolor='#E0E0E0')

        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>{timestamp}</sub>",
                x=0.5,
                xanchor='center',
                font=dict(size=18, family='Arial Bold')
            ),
            height=600,
            width=1200,
            showlegend=True,
            template="plotly_white",
            hovermode='closest'
        )

        return fig

    def create_liquidity_chart(self,
                              bins_data: pd.DataFrame,
                              price_col: str = 'price',
                              liquidity_col: str = 'total_value',
                              bin_id_col: str = 'binId',
                              current_price: Optional[float] = None,
                              active_bin_id: Optional[int] = None,
                              title: str = "Liquidity Distribution") -> go.Figure:
        """Create a liquidity distribution chart"""

        if bins_data.empty:
            logger.warning("No bin data available for chart creation")
            return None

        # Create figure
        fig = go.Figure()

        # Color bins based on active bin
        colors = ['red' if (active_bin_id and bid < active_bin_id)
                 else 'green' if (active_bin_id and bid > active_bin_id)
                 else 'yellow' for bid in bins_data[bin_id_col]]

        # Main liquidity chart
        fig.add_trace(
            go.Bar(
                x=bins_data[price_col],
                y=bins_data[liquidity_col],
                name='Liquidity',
                marker=dict(color=colors, opacity=0.7),
                hovertemplate=(
                    f'Price: %{{x:.8f}}<br>'
                    f'Liquidity: %{{y:,.0f}}<br>'
                    f'Bin ID: %{{customdata}}<br>'
                    '<extra></extra>'
                ),
                customdata=bins_data[bin_id_col]
            )
        )

        # Add current price line if provided
        if current_price:
            fig.add_vline(
                x=current_price,
                line_dash="dash",
                line_color="blue",
                line_width=2,
                annotation_text=f"Current Price: {self.format_price(current_price)}",
                annotation_position="top"
            )

        # Update layout
        fig.update_xaxes(title_text="Price", showgrid=True, gridwidth=1, gridcolor='#E0E0E0')
        fig.update_yaxes(title_text="Liquidity")

        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>{timestamp}</sub>",
                x=0.5,
                xanchor='center',
                font=dict(size=18, family='Arial Bold')
            ),
            height=600,
            width=1200,
            showlegend=True,
            template="plotly_white",
            hovermode='x unified'
        )

        return fig

    def save_chart(self, fig: go.Figure, filename: str, format: str = "png") -> bool:
        """Save chart to file"""
        try:
            if format.lower() == "png":
                fig.write_image(filename, width=1200, height=800, scale=2)
            elif format.lower() == "html":
                fig.write_html(filename)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Chart saved: {filename}")
            return True

        except ImportError as e:
            if "kaleido" in str(e).lower():
                logger.warning("Kaleido not installed - saving as HTML instead")
                html_filename = filename.replace('.png', '.html')
                fig.write_html(html_filename)
                logger.info(f"Chart saved as HTML: {html_filename}")
                return True
            else:
                logger.error(f"Import error saving chart: {e}")
                return False

        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return False

    def calculate_price_range_distribution(self,
                                         bins_data: pd.DataFrame,
                                         current_price: float,
                                         price_col: str = 'price',
                                         liquidity_col: str = 'total_value',
                                         ranges: List[float] = [5, 10, 15, 20]) -> Dict[float, float]:
        """Calculate liquidity distribution at different price ranges from current price"""

        total_liquidity = bins_data[liquidity_col].sum()
        if total_liquidity == 0:
            return {pct: 0.0 for pct in ranges}

        distributions = {}

        for pct in ranges:
            lower_price = current_price * (1 - pct/100)
            upper_price = current_price * (1 + pct/100)

            bins_in_range = bins_data[
                (bins_data[price_col] >= lower_price) &
                (bins_data[price_col] <= upper_price)
            ]

            liquidity_in_range = bins_in_range[liquidity_col].sum() if not bins_in_range.empty else 0
            pct_of_total = (liquidity_in_range / total_liquidity * 100) if total_liquidity > 0 else 0

            distributions[pct] = pct_of_total

        return distributions

    def export_to_csv(self,
                      data: pd.DataFrame,
                      filename: str,
                      metadata: Optional[Dict] = None) -> bool:
        """Export DataFrame to CSV with optional metadata"""
        try:
            # Add metadata columns if provided
            if metadata:
                for key, value in metadata.items():
                    if key not in data.columns:
                        data[key] = value

            # Add export timestamp
            data['export_timestamp'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

            # Export to CSV
            data.to_csv(filename, index=False, float_format='%.8f')

            logger.info(f"Data exported to CSV: {filename}")
            return True

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def clear_token_cache(self):
        """Clear the token information cache"""
        self.token_cache.clear()
        logger.info("Token cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'token_cache_size': len(self.token_cache),
            'cached_tokens': len([k for k, v in self.token_cache.items() if v is not None]),
            'failed_lookups': len([k for k, v in self.token_cache.items() if v is None])
        }