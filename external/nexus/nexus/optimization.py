from sklearn.model_selection import ParameterGrid
class StrategyOptimizer:
    """Optimizes a strategy using grid search."""

    def __init__(self, strategy_class, param_grid, backtest_params, run_backtest_func):
        self.strategy_class = strategy_class
        self.param_grid = list(ParameterGrid(param_grid))
        self.backtest_params = backtest_params
        self.run_backtest_func = run_backtest_func

    def optimize(self):
        results = []
        for params in self.param_grid:
            performance = self.run_backtest_func(params)
            results.append((params, performance))
        # Sort results by performance metric in descending order
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def evaluate_performance(self, portfolio):
        # Simple performance metric: final portfolio value
        return portfolio.all_holdings[-1]['total']