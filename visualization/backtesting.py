from decimal import Decimal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from hummingbot.connector.connector_base import TradeType
from plotly.subplots import make_subplots

from visualization import theme
from visualization.theme import get_default_layout


def add_executors_trace(fig, executors, row, col):
    for executor in executors:
        entry_time = pd.to_datetime(executor.timestamp, unit='s')
        entry_price = executor.custom_info["current_position_average_price"]
        exit_time = pd.to_datetime(executor.close_timestamp, unit='s')
        exit_price = executor.custom_info["close_price"]
        name = "Buy Executor" if executor.config.side == TradeType.BUY else "Sell Executor"

        if executor.filled_amount_quote == 0:
            fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, entry_price], mode='lines',
                                     line=dict(color='grey', width=2, dash="dash"), name=name), row=row, col=col)
        else:
            if executor.net_pnl_quote > Decimal(0):
                fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                         line=dict(color='green', width=3), name=name), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(x=[entry_time, exit_time], y=[entry_price, exit_price], mode='lines',
                                         line=dict(color='red', width=3), name=name), row=row, col=col)
    return fig


def get_pnl_trace(executors):
    pnl = [e.net_pnl_quote for e in executors]
    cum_pnl = np.cumsum(pnl)
    return go.Scatter(
        x=pd.to_datetime([e.close_timestamp for e in executors], unit="s"),
        y=cum_pnl,
        mode='lines',
        line=dict(color='gold', width=2, dash="dash"),
        name='Cumulative PNL'
    )


def get_bt_candlestick_trace(df):
    df.index = pd.to_datetime(df.timestamp, unit='s')
    return go.Scatter(x=df.index,
                      y=df['close'],
                      mode='lines',
                      line=dict(color=theme.get_color_scheme()["price"]),
                      )


def create_backtesting_figure(df, executors, config, summary_results):
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, subplot_titles=('Candlestick', 'PNL Quote'),
                        row_heights=[0.8, 0.2])

    # Add candlestick trace
    fig.add_trace(get_bt_candlestick_trace(df), row=1, col=1)

    # Add executors trace
    fig = add_executors_trace(fig, executors, row=1, col=1)

    # Add PNL trace
    fig.add_trace(get_pnl_trace(executors), row=2, col=1)

    # Apply the theme layout
    layout_settings = get_default_layout(f"Trading Pair: {config['trading_pair']}")
    layout_settings["showlegend"] = False
    fig.update_layout(**layout_settings)

    # Update axis properties
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="PNL", row=2, col=1)

    # Add annotations for backtesting metrics
    max_drawdown_usd = summary_results.get('max_drawdown_usd', 0)
    max_drawdown_pct = summary_results.get('max_drawdown_pct', 0)
    metrics = [
        f"Net PNL (Quote): {summary_results.get('net_pnl_quote', 0):.2f} ({summary_results.get('net_pnl', 0):.2%})",
        f"Max Drawdown (USD): {max_drawdown_usd:.2f} ({max_drawdown_pct:.2%})",
        f"Total Volume (Quote): {summary_results.get('total_volume', 0):.2f}",
        f"Sharpe Ratio: {summary_results.get('sharpe_ratio', 0):.2f}",
        f"Profit Factor: {summary_results.get('profit_factor', 0):.2f}",
        f"Total Executors with Position: {summary_results.get('total_executors_with_position', 0)}",
        f"Global Accuracy: {summary_results.get('accuracy', 0):.2%}",
        f"Total Long: {summary_results.get('total_long', 0)}",
        f"Total Short: {summary_results.get('total_short', 0)}",
        f"Accuracy Long: {summary_results.get('accuracy_long', 0):.2%}",
        f"Accuracy Short: {summary_results.get('accuracy_short', 0):.2%}",
        f"TAKE PROFIT: {summary_results.get('close_types', {}).get('TAKE_PROFIT', 0)}",
        f"TRAILING STOP: {summary_results.get('close_types', {}).get('TRAILING_STOP', 0)}",
        f"STOP LOSS: {summary_results.get('close_types', {}).get('STOP_LOSS', 0)}",
        f"TIME LIMIT: {summary_results.get('close_types', {}).get('TIME_LIMIT', 0)}",
        f"EARLY STOP: {summary_results.get('close_types', {}).get('EARLY_STOP', 0)}",
    ]

    annotations = []
    for i, metric in enumerate(metrics):
        annotations.append(dict(
            xref='paper', yref='paper',
            x=0.8, y=1.0 - i * 0.02,
            xanchor='left', yanchor='middle',
            text=metric,
            showarrow=False,
            font=dict(size=10)
        ))

    fig.update_layout(annotations=annotations)
    return fig
