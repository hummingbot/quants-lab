from decimal import Decimal
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data_structures.data_structure_base import DataStructureBase
from hummingbot.connector.connector_base import TradeType
from hummingbot.strategy_v2.controllers import ControllerConfigBase


class BacktestingResult(DataStructureBase):
    def __init__(self, backtesting_result: Dict, controller_config: ControllerConfigBase):
        super().__init__(backtesting_result)
        self.processed_data = backtesting_result["processed_data"]["features"]
        self.results = backtesting_result["results"]
        self.executors = backtesting_result["executors"]
        self.controller_config = controller_config

    def get_results_summary(self, results: Optional[Dict] = None):
        if results is None:
            results = self.results
        net_pnl_quote = results["net_pnl_quote"]
        net_pnl_pct = results["net_pnl"]
        max_drawdown = results["max_drawdown_usd"]
        max_drawdown_pct = results["max_drawdown_pct"]
        total_volume = results["total_volume"]
        sharpe_ratio = results["sharpe_ratio"]
        profit_factor = results["profit_factor"]
        total_executors = results["total_executors"]
        accuracy_long = results["accuracy_long"]
        accuracy_short = results["accuracy_short"]
        take_profit = results["close_types"].get("TAKE_PROFIT", 0)
        stop_loss = results["close_types"].get("STOP_LOSS", 0)
        time_limit = results["close_types"].get("TIME_LIMIT", 0)
        trailing_stop = results["close_types"].get("TRAILING_STOP", 0)
        early_stop = results["close_types"].get("EARLY_STOP", 0)
        return f"""
Net PNL: ${net_pnl_quote:.2f} ({net_pnl_pct*100:.2f}%) | Max Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct*100:.2f}%)
Total Volume ($): {total_volume:.2f} | Sharpe Ratio: {sharpe_ratio:.2f} | Profit Factor: {profit_factor:.2f}
Total Executors: {total_executors} | Accuracy Long: {accuracy_long:.2f} | Accuracy Short: {accuracy_short:.2f}
Close Types: Take Profit: {take_profit} | Stop Loss: {stop_loss} | Time Limit: {time_limit} |
             Trailing Stop: {trailing_stop} | Early Stop: {early_stop}
"""

    @property
    def executors_df(self):
        executors_df = pd.DataFrame([e.dict() for e in self.executors])
        executors_df["side"] = executors_df["config"].apply(lambda x: x["side"].name)
        return executors_df

    def _get_bt_candlestick_trace(self):
        self.processed_data.index = pd.to_datetime(self.processed_data.timestamp, unit='s')
        return go.Scatter(x=self.processed_data.index,
                          y=self.processed_data['close'],
                          mode='lines',
                          line=dict(color="blue"),
                          )

    @staticmethod
    def _get_pnl_trace(executors, line_style: str = "dash"):
        pnl = [e.net_pnl_quote for e in executors]
        cum_pnl = np.cumsum(pnl)
        return go.Scatter(
            x=pd.to_datetime([e.close_timestamp for e in executors], unit="s"),
            y=cum_pnl,
            mode='lines',
            line=dict(color='gold', width=2, dash=line_style if line_style == "dash" else None),
            name='Cumulative PNL'
        )

    @staticmethod
    def _get_default_layout(title=None, height=800, width=1200):
        layout = {
            "template": "plotly_dark",
            "plot_bgcolor": 'rgba(0, 0, 0, 0)',  # Transparent background
            "paper_bgcolor": 'rgba(0, 0, 0, 0.1)',  # Lighter shade for the paper
            "font": {"color": 'white', "size": 12},  # Consistent font color and size
            "height": height,
            "width": width,
            "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
            "xaxis_rangeslider_visible": False,
            "hovermode": "x unified",
            "showlegend": False,
        }
        if title:
            layout["title"] = title
        return layout

    @staticmethod
    def _add_executors_trace(fig, executors, row=1, col=1, line_style="dash"):
        for executor in executors:
            entry_time = pd.to_datetime(executor.timestamp, unit='s')
            entry_price = executor.custom_info["current_position_average_price"]
            exit_time = pd.to_datetime(executor.close_timestamp, unit='s')
            exit_price = executor.custom_info["close_price"]
            name = "Buy Executor" if executor.config.side == TradeType.BUY else "Sell Executor"

            if executor.filled_amount_quote == 0:
                fig.add_trace(
                    go.Scatter(x=[entry_time, exit_time], y=[entry_price, entry_price], mode='lines', showlegend=False,
                               line=dict(color='grey', width=2, dash=line_style if line_style == "dash" else None),
                               name=name), row=row, col=col)
            else:
                if executor.net_pnl_quote > Decimal(0):
                    fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=False,
                                             line=dict(color='green', width=2,
                                                       dash=line_style if line_style == "dash" else None), name=name),
                                  row=row,
                                  col=col)
                else:
                    fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                             showlegend=False,
                                             line=dict(color='red', width=2,
                                                       dash=line_style if line_style == "dash" else None), name=name),
                                  row=row, col=col)

        return fig

    def get_backtesting_figure(self):
        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02, subplot_titles=('Candlestick', 'PNL Quote'),
                            row_heights=[0.7, 0.3])

        # Add candlestick trace
        fig.add_trace(self._get_bt_candlestick_trace(), row=1, col=1)

        # Add executors trace
        fig = self._add_executors_trace(fig, self.executors, row=1, col=1)

        # Add PNL trace
        fig.add_trace(self._get_pnl_trace(self.executors), row=2, col=1)

        # Apply the theme layout
        layout_settings = self._get_default_layout(f"Trading Pair: {self.controller_config.dict().get('trading_pair')}")
        layout_settings["showlegend"] = False
        fig.update_layout(**layout_settings)

        # Update axis properties
        fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
        fig.update_xaxes(row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="PNL", row=2, col=1)
        return fig
