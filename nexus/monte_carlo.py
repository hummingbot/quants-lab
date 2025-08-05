import numpy as np
import pandas as pd

class MonteCarloAnalyzer:
    """Performs Monte Carlo analysis on an equity curve."""

    def __init__(self, equity_curve):
        self.equity_curve = equity_curve

    def resample_equity_curve(self, num_simulations=1000):
        returns = self.equity_curve['returns']
        simulations = []
        for _ in range(num_simulations):
            resampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            simulation = (1 + pd.Series(resampled_returns)).cumprod()
            simulations.append(simulation)
        return simulations

    def analyze(self, num_simulations=1000):
        simulations = self.resample_equity_curve(num_simulations)
        simulation_df = pd.concat(simulations, axis=1)
        return simulation_df
