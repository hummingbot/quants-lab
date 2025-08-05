import multiprocessing
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from nexus.backtest import Backtest
from nexus.charting import show_plotly_figure
from nexus.execution import SimulatedExecutionHandler
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from nexus.performance import PerformanceReport
from nexus.portfolio import NaivePortfolio
from strategy.ma_cross import MovingAverageCrossStrategy


def main():
    # Define the parameter grid
    param_grid = {
        "short_window": [5, 10, 20, 30, 40],
        "long_window": [50, 100, 150, 200, 250],
    }

    # Convert parameter grid to a list of dictionaries
    param_list = list(ParameterGrid(param_grid))

    # Backtest parameters
    backtest_params = {
        "symbols": ["AAPL"],
        "history_dir": "history/yf",
        "initial_capital": 100000.0,
        "start_date": "2013-01-01",
        "end_date": "2023-01-01",
    }

    # Use multiprocessing Pool to run backtests in parallel
    # Set to the desired number of processes if needed
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        # Map the worker function to the parameter list
        results_list = list(
            tqdm(
                pool.imap_unordered(
                    run_backtest_wrapper,
                    [(params, backtest_params) for params in param_list],
                ),
                total=len(param_list),
            )
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)

    # Print results
    print("Optimization Results:")
    print(results_df)

    log_dir = os.path.join("log", "optimize")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Optimization reports saved here: {log_dir}\n")

    # Save results to CSV (optional)
    optimization_results = os.path.join(log_dir, "optimization_results.csv")
    results_df.to_csv(optimization_results, index=False)
    # Plot the results
    plot_optimization_results(results_df, log_dir)


def run_backtest_wrapper(args):
    params, backtest_params = args
    return run(params, backtest_params)


def run(params, backtest_params):
    """
    Runs a single backtest with the given parameters and returns the performance metric.

    Parameters:
    - params (dict): Dictionary of strategy parameters (e.g., {'short_window': 10, 'long_window': 50})
    - backtest_params (dict): Dictionary of backtest parameters

    Returns:
    - result (dict): Dictionary containing the parameters and performance metric
    """
    try:
        # Initialize the backtest
        backtest = Backtest(
            backtest_params=backtest_params,
            strategy_class=MovingAverageCrossStrategy,
            strategy_params=params,
            data_handler_class=HistoricCSVDataHandler,
            execution_handler_class=SimulatedExecutionHandler,
            portfolio_class=NaivePortfolio,
            reporting=False,  # Disable reporting during optimization
        )

        # Run the backtest
        backtest.run()

        # Generate performance report
        report = PerformanceReport(
            data_handler=backtest.data_handler,
            portfolio=backtest.portfolio,
            execution_time=backtest.execution_time,
        )
        performance = report.generate_report(symbol=backtest_params["symbols"][0])
        sharpe_ratio = performance.get("Sharpe Ratio", 0.0)

        # Return results as a dictionary
        result = {
            "short_window": params["short_window"],
            "long_window": params["long_window"],
            "sharpe_ratio": sharpe_ratio,
        }
        return result

    except Exception as e:
        # Handle exceptions and return None or a default value for the performance metric
        print(f"Error with params {params}: {e}")
        return {
            "short_window": params.get("short_window"),
            "long_window": params.get("long_window"),
            "sharpe_ratio": None,  # or set to 0.0 if you prefer
        }


def plot_optimization_results(results_df, log_dir):
    # Pivot the DataFrame
    pivot_table = results_df.pivot(
        index="short_window", columns="long_window", values="sharpe_ratio"
    )
    pivot_table = pivot_table.sort_index(ascending=True)
    pivot_table = pivot_table.sort_index(axis=1, ascending=True)

    # Get the parameter values
    short_windows = pivot_table.index.values
    long_windows = pivot_table.columns.values
    sharpe_ratios = pivot_table.values

    # Plot Heatmap
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            x=long_windows,
            y=short_windows,
            z=sharpe_ratios,
            colorscale="Viridis",
            colorbar=dict(title="Sharpe Ratio"),
        )
    )

    fig_heatmap.update_layout(
        title="Optimization Heatmap: Sharpe Ratio",
        xaxis_title="Long Window",
        yaxis_title="Short Window",
    )

    fig_heatmap.write_html(os.path.join(log_dir, "optimization_heatmap.html"))
    # Optionally display the plot
    show_plotly_figure(
        fig_heatmap, base_name="optimization_heatmap"
    )  # WSL workaround, instead of fig.show()

    # Plot Surface
    X, Y = np.meshgrid(long_windows, short_windows)
    Z = sharpe_ratios

    fig_surface = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Viridis")])

    fig_surface.update_layout(
        title="Optimization Surface Plot: Sharpe Ratio",
        scene=dict(
            xaxis_title="Long Window",
            yaxis_title="Short Window",
            zaxis_title="Sharpe Ratio",
        ),
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    fig_surface.write_html(os.path.join(log_dir, "optimization_surface.html"))
    # Optionally display the plot
    show_plotly_figure(
        fig_surface, base_name="optimization_surface"
    )  # WSL workaround, instead of fig.show()


if __name__ == "__main__":
    main()
