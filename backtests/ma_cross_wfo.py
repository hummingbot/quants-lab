import pandas as pd
import multiprocessing
from datetime import timedelta
import queue
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from nexus.backtest import Backtest
from nexus.feed.csv_ohlc import HistoricCSVDataHandler
from nexus.execution import SimulatedExecutionHandler
from nexus.portfolio import NaivePortfolio
from nexus.performance import PerformanceReport
from strategy.ma_cross import MovingAverageCrossStrategy

def main():
    # WFO parameters
    num_cycles = 5
    training_size = timedelta(days=365 * 2)  # 2 years
    testing_size = timedelta(days=365)       # 1 year
    anchored = False

    # Parameter grid
    param_grid = {
        'short_window': [10, 20, 30, 40],
        'long_window': [50, 100, 200, 250]
    }

    # Backtest parameters
    history_files = {'AAPL': 'data/AAPL.csv'}
    symbol_list = ['AAPL']
    initial_capital = 100000.0

    # Load data to get date range
    data = pd.read_csv(history_files['AAPL'], parse_dates=True, index_col='Date')
    dates = data.index

    # Calculate cycles
    cycle_dates = calculate_cycles(dates, num_cycles, training_size, testing_size, anchored)

    all_results = []

    for i, cycle in enumerate(cycle_dates):
        print(f"\nCycle {i+1}/{len(cycle_dates)}:")
        print(f"Training from {cycle['train_start'].date()} to {cycle['train_end'].date()}")
        print(f"Testing from {cycle['test_start'].date()} to {cycle['test_end'].date()}")

        # Optimize on training data
        best_params = optimize_on_training_data(cycle['train_start'], cycle['train_end'], param_grid, history_files, symbol_list, initial_capital)

        if best_params:
            print(f"Best Parameters: {best_params}")
            # Backtest on testing data
            performance = backtest_on_testing_data(cycle['test_start'], cycle['test_end'], best_params, history_files, symbol_list, initial_capital)
        else:
            print("No valid parameters found.")
            performance = None

        all_results.append({
            'cycle': i+1,
            'train_start': cycle['train_start'],
            'train_end': cycle['train_end'],
            'test_start': cycle['test_start'],
            'test_end': cycle['test_end'],
            'best_params': best_params,
            'performance': performance
        })

    # Analyze results
    analyze_wfo_results(all_results)

def calculate_cycles(dates, num_cycles, training_size, testing_size, anchored):
    cycles = []
    start_date = dates.min()
    end_date = dates.max()
    current_train_start = start_date

    for _ in range(num_cycles):
        if anchored and cycles:
            train_start = start_date
        else:
            train_start = current_train_start

        train_end = train_start + training_size
        test_start = train_end
        test_end = test_start + testing_size

        if test_end > end_date:
            test_end = end_date

        if train_end >= end_date:
            break

        cycles.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })

        current_train_start += testing_size

        if test_end == end_date:
            break

    return cycles

def optimize_on_training_data(train_start, train_end, param_grid, history_files, symbol_list, initial_capital):
    backtest_params = {
        'history_files': history_files,
        'symbol_list': symbol_list,
        'initial_capital': initial_capital,
        'data_handler_class': HistoricCSVDataHandler,
        'execution_handler_class': SimulatedExecutionHandler,
        'portfolio_class': NaivePortfolio,
        'start_date': train_start,
        'end_date': train_end
    }

    param_list = list(ParameterGrid(param_grid))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        results = list(tqdm(pool.imap_unordered(run_backtest_wrapper, [(params, backtest_params) for params in param_list]), total=len(param_list)))
    results_df = pd.DataFrame(results)
    results_df.dropna(subset=['sharpe_ratio'], inplace=True)

    if results_df.empty:
        return None

    best_row = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    best_params = {'short_window': int(best_row['short_window']), 'long_window': int(best_row['long_window'])}
    return best_params

def run_backtest_wrapper(args):
    params, backtest_params = args
    return run(params, backtest_params)

def run(params, backtest_params):
    try:
        events = queue.Queue()
        backtest = Backtest(
            history_dir=backtest_params['history_files'],
            symbols=backtest_params['symbol_list'],
            initial_capital=backtest_params['initial_capital'],
            strategy_class=MovingAverageCrossStrategy,
            strategy_params=params,
            data_handler_class=backtest_params['data_handler_class'],
            execution_handler_class=backtest_params['execution_handler_class'],
            portfolio_class=backtest_params['portfolio_class'],
            events=events,
            reporting=False,
            start_date=backtest_params['start_date'],
            end_date=backtest_params['end_date']
        )
        backtest.run()
        report = PerformanceReport(backtest.portfolio)
        performance = report.generate_report()
        sharpe_ratio = performance.get('Sharpe Ratio', 0.0)
        return {'short_window': params['short_window'], 'long_window': params['long_window'], 'sharpe_ratio': sharpe_ratio}
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return {'short_window': params.get('short_window'), 'long_window': params.get('long_window'), 'sharpe_ratio': None}

def backtest_on_testing_data(test_start, test_end, best_params, history_files, symbol_list, initial_capital):
    try:
        events = queue.Queue()
        backtest = Backtest(
            history_dir=history_files,
            symbols=symbol_list,
            initial_capital=initial_capital,
            strategy_class=MovingAverageCrossStrategy,
            strategy_params=best_params,
            data_handler_class=HistoricCSVDataHandler,
            execution_handler_class=SimulatedExecutionHandler,
            portfolio_class=NaivePortfolio,
            events=events,
            reporting=False,
            start_date=test_start,
            end_date=test_end
        )
        backtest.run()
        report = PerformanceReport(backtest.portfolio)
        performance = report.generate_report()
        print("Testing Performance:")
        for key, value in performance.items():
            print(f"{key}: {value}")
        return performance
    except Exception as e:
        print(f"Error during testing with params {best_params}: {e}")
        return None

def analyze_wfo_results(all_results):
    # Aggregate and analyze the results
    pass

if __name__ == '__main__':
    main()
