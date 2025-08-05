import pathlib
import subprocess
import uuid
from enum import Enum

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def show_plotly_figure(fig, base_name: str = "plot"):
    """
    Save a Plotly figure to /tmp/<base_name>_<uid>.html inside WSL
    and open it in the default Windows browser (WSL2-compatible).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to display.
    base_name : str, optional
        Base part of the filename before the UID (default: 'plot').
    # ── Example usage ─────────────────────────────────────────
    if __name__ == "__main__":
    fig = go.Figure(data=[go.Bar(y=[5, 3, 6])])
    show_plotly_figure(fig, base_name="sales")
    """
    # 1️⃣  Ensure /tmp exists and build a unique filename
    tmp_dir = pathlib.Path("/tmp")
    tmp_dir.mkdir(exist_ok=True)  # /tmp normally exists, but safe to call
    uid = uuid.uuid4().hex[:8]  # 8-char random tag
    filename = f"{base_name}_{uid}.html"
    html_path = (tmp_dir / filename).resolve()

    # 2️⃣  Write the HTML
    fig.write_html(str(html_path), auto_open=False)

    try:
        # 3️⃣  Convert to Windows path
        windows_path = (
            subprocess.check_output(["wslpath", "-w", str(html_path)]).decode().strip()
        )

        # 4️⃣  Launch in Windows (safe CWD to mute UNC warning)
        subprocess.run(
            ["cmd.exe", "/C", "start", "", windows_path],
            cwd="/mnt/c",
            check=False,
        )
    except Exception as e:
        print(f"[Plotly] Could not open in browser: {e}")
        print(f"[Plotly] You can still open the file manually: {html_path}")


class PlotLocation(Enum):
    MAIN = "MAIN"
    NEW = "NEW"


def plot_monte_carlo(simulations):
    fig = go.Figure()
    for simulation in simulations.columns:
        fig.add_trace(
            go.Scatter(
                x=simulations.index,
                y=simulations[simulation],
                mode="lines",
                line=dict(color="rgba(0,0,255,0.1)"),
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Monte Carlo Simulations",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
    )
    show_plotly_figure(fig)  # WSL workaround, instead of fig.show()


def plot_equity_curve(equity_curve, initial_capital, save_path, show_results=False):
    fig = go.Figure()
    equity = equity_curve["total"]  # - initial_capital
    # Plot equity curve
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            fill="tozeroy",
            fillcolor="rgba(0, 0, 255, 0.3)",
            line=dict(color="blue"),
            name="Equity Curve",
        )
    )

    # Calculate and plot drawdown
    max_equity = equity.cummax()
    drawdown = equity - max_equity

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=drawdown,
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red"),
            name="Drawdown",
        )
    )

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
    )

    # Save the plot to an HTML file
    fig.write_html(save_path)

    if show_results:
        show_plotly_figure(fig)  # WSL workaround, instead of fig.show()


def plot_performance(symbol, data, trades, performance):
    """
    Combines price and indicators plot with equity curve and drawdown into a single figure with two subplots.

    Parameters:
    - symbol (str): The symbol to plot.
    - data (pd.DataFrame): The price data with a datetime index.
    - equity_curve (pd.DataFrame): The equity curve DataFrame with a datetime index.
    - trades (pd.DataFrame): The trades DataFrame with entry and exit points.
    - performance (report as dict)
    """
    # Create a subplot layout with 1 row and 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
        horizontal_spacing=0.05,  # Adjust spacing between columns
        column_widths=[
            0.2,
            0.8,
        ],  # Set the table column to 20% width and plot column to 80%
        specs=[
            [{"type": "table"}, {"secondary_y": True}]  # Single row configuration
        ],
    )

    # Add the performance table to the first column

    # Round numerical values to 2 decimal places
    formatted_performance = {
        k: round(v, 2) if isinstance(v, (float, int)) else v
        for k, v in performance.items()
    }

    # Prepare headers and values for the table
    headers = list(formatted_performance.keys())
    values = list(formatted_performance.values())

    # Add table to the left column
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="paleturquoise",
                align=["right", "left"],
                font=dict(size=10, color="black"),
            ),
            cells=dict(
                values=[headers, values],
                fill_color="lavender",
                align=[
                    "right",
                    "left",
                ],  # Align 'Metric' cells to the right and 'Value' cells to the leftalign='left',
                font=dict(size=10, color="black"),
            ),
        ),
        row=1,
        col=1,
    )

    # Add the combined Equity, Drawdown, and Price plot to the right column
    # Plot price chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color="black"),
        ),
        row=1,
        col=2,
    )

    # Plot buy/sell signals on price chart
    if not trades.empty:
        buy_signals = trades[trades["direction"] == "BUY"]
        sell_signals = trades[trades["direction"] == "SELL"]

        fig.add_trace(
            go.Scatter(
                x=buy_signals["datetime"],
                y=buy_signals["price"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=10),
                name="Buy",
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=sell_signals["datetime"],
                y=sell_signals["price"],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=10),
                name="Sell",
            ),
            row=1,
            col=2,
        )

    # Plot equity curve
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[symbol],
            fill="tozeroy",
            fillcolor="rgba(0, 0, 255, 0.3)",
            line=dict(color="blue"),
            name="Equity",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    """
    # Plot equity curve as bar plot
    fig.add_trace(
        go.Bar(
            x=data.index,  # x-axis values
            y=data[symbol],  # y-axis values corresponding to the equity data
            marker=dict(color='blue', opacity=0.6),  # Set bar color and opacity
            name='Equity'
        ),
        row=1, col=2, secondary_y=True
    )
    """
    # Calculate and plot drawdown
    equity = data[symbol]
    drawdown = equity - equity.cummax()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=drawdown,
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)",
            line=dict(color="red"),
            name="Drawdown",
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title=f"{symbol} strategy backtest performance",
        # xaxis_title="Date",
        # yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="seaborn",
        hovermode="x unified",
    )

    # Update y-axis titles for clarity
    fig.update_yaxes(title_text="Price", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Equity / Drawdown", row=1, col=2, secondary_y=True)

    return fig


def plot_indicators(symbol, data, trades, indicators, plot_locations):
    """
    Combines price and indicators plot with equity curve and drawdown into a single figure with two subplots.

    Parameters:
    - symbol (str): The symbol to plot.
    - data (pd.DataFrame): The price data with a datetime index.
    - trades (pd.DataFrame): The trades DataFrame with entry and exit points.
    - indicators (dict): A dictionary of indicators to plot. Keys are indicator names, values are numpay array.
    - plot_locations (dict): Dictionary mapping indicator names to plot locations ('MAIN' or 'NEW')."""

    # Identify indicators to plot on MAIN and NEW
    main_indicators = {
        k: v for k, v in indicators.items() if plot_locations.get(k, "MAIN") == "MAIN"
    }
    new_indicators = {
        k: v for k, v in indicators.items() if plot_locations.get(k, "NEW") == "NEW"
    }

    # Determine the number of subplots: minmum 1
    rows = 1 + len(new_indicators)

    # Create a figure with  shared x-axis
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02 * rows,
    )  # Use dynamic specs for subplot types

    # Plot price chart
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            name="Price",
            line=dict(color="black"),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)

    # Plot buy/sell signals on price chart
    if not trades.empty:
        buy_signals = trades[trades["direction"] == "BUY"]
        sell_signals = trades[trades["direction"] == "SELL"]

        fig.add_trace(
            go.Scatter(
                x=buy_signals["datetime"],
                y=buy_signals["price"],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=10),
                name="Buy",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=sell_signals["datetime"],
                y=sell_signals["price"],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=10),
                name="Sell",
            ),
            row=1,
            col=1,
        )

    for indicator_name, indicator_series in main_indicators.items():
        fig.add_trace(
            go.Scatter(
                x=indicator_series.index,
                y=indicator_series.values,
                mode="lines",
                name=indicator_name,
            ),
            row=1,
            col=1,
        )

    # Plot new indicators in separate subplots
    current_row = 2
    for indicator_name, indicator_series in new_indicators.items():
        fig.add_trace(
            go.Scatter(
                x=indicator_series.index,
                y=indicator_series.values,
                mode="lines",
                name=indicator_name,
            ),
            row=current_row,
            col=1,
        )
        fig.update_yaxes(title_text=indicator_name, row=current_row, col=1)
        current_row += 1

    # Update layout
    fig.update_layout(
        title=f"{symbol} strategy indicators",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="seaborn",
        hovermode="x unified",
    )

    return fig
