import queue
import os
import time
import pandas as pd
import logging
from nexus.event import OrderEvent
from nexus.logger import setup_logging
from nexus.performance import PerformanceReport
from nexus.charting import plot_performance, plot_indicators, plot_equity_curve, show_plotly_figure, PlotLocation
from nexus.asset import load_assets, Asset

class Backtest:
    """
    Encapsulates the settings and components for backtesting.
    """
    def __init__(self, 
                 backtest_params, 
                 strategy_class, 
                 strategy_params=None,
                 data_handler_class=None,
                 execution_handler_class=None,
                 portfolio_class=None, 
                 events=None, 
                 reporting=True, 
                 log_level=logging.INFO,
                 open_in_browser = True): # Open plots in the default browser automatically
        
        self.strategy_name = strategy_class.__name__       
        setup_logging(self.strategy_name, log_level, log_to_file=reporting)
        self.logger = logging.getLogger(self.__class__.__name__)  
        self.logger.debug("__init__ class Backtest")         
        self.logger.info(f"Backtest period: {backtest_params['start_date']} to {backtest_params['end_date']}")

        if strategy_params is None:
            strategy_params = {}

        # Extract backtest parameters
        self.logger.debug(f"backtest_params: {backtest_params}")  
        self.symbols = backtest_params.get('symbols', [])
        self.history_dir = backtest_params.get('history_dir', 'history')
        self.aggregation = backtest_params.get('aggregation', 'h')
        self.start_date = backtest_params.get('start_date', None)
        self.end_date = backtest_params.get('end_date', None)
        self.look_back = backtest_params.get('look_back', 80)
        self.initial_capital = backtest_params.get('initial_capital', 0.0)
        self.max_long = backtest_params.get('max_long', -1)
        self.max_short = backtest_params.get('max_short', -1)
        self.events = events if events else queue.Queue()
        self.reporting = reporting
        self.execution_time = None
        self.open_in_browser = open_in_browser

        # Load assets from assets.csv
        self.logger.debug(f"Loading file assets.csv from: {self.history_dir}...")
        assets_csv_path = os.path.join(self.history_dir, 'assets.csv')
        if os.path.exists(assets_csv_path):
            self.assets = load_assets(assets_csv_path)
        else:
            # Create basic assets on the fly
            self.logger.warning("assets.csv not found, using default assets")
            self.assets = {
                sym: Asset(
                    name=sym,
                    price=0.0,
                    spread=0.0,
                    roll_long=0.0,
                    roll_short=0.0,
                    pip=1.0,
                    pip_cost=1.0,
                    margin_cost=0.0,
                    market='',
                    lot_amount=1.0,
                    commission=0.0,
                    symbol=sym,
                    asset_type='forex'
                )
                for sym in self.symbols
            }

        # Validate symbols
        for symbol in self.symbols:
            if symbol not in self.assets:
                raise ValueError(f"Symbol '{symbol}' not found in assets configuration.")

        # Initialize data handler with start_date and end_date
        self.data_handler = data_handler_class(
            events =self.events, 
            symbols =self.symbols,
            history_dir = self.history_dir,
            start_date =self.start_date, 
            end_date =self.end_date, 
            aggregation = self.aggregation
        )

        # Initialize strategy
        self.strategy = strategy_class(
            events=self.events, 
            assets=self.assets, 
            strategy_params=strategy_params,
            backtest_params= backtest_params
        )
        
        # Initialize execution handler
        self.execution_handler = execution_handler_class(
            events=self.events, 
            data_handler=self.data_handler, 
            assets=self.assets
        )

        # Initialize portfolio
        self.portfolio = portfolio_class(
            events=self.events, 
            data_handler=self.data_handler, 
            symbols=self.symbols, 
            initial_capital=self.initial_capital, 
            strategy_name=self.strategy_name,
            assets=self.assets
        )

    def run(self):
        """
        Executes the backtest by running the event-driven simulation.
        """
        start_time = time.time()
        while self.data_handler.continue_backtest:
            self.data_handler.get_next_bar()
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    if event.type == 'MARKET':
                        self.strategy.run(event)
                        self.portfolio.update_time_index(event)
                        # Check for stop losses after processing signals
                        self.portfolio.update_market(event)
                        # Process pending limit orders
                        self.execution_handler.update_market(event)
                    elif event.type == 'SIGNAL':
                        self.portfolio.update_signal(event)
                    elif event.type == 'ORDER':
                        self.execution_handler.execute_order(event)
                    elif event.type == 'FILL':
                        self.portfolio.update_fill(event)

        self.close_all_positions()
        end_time = time.time()
        self.execution_time = end_time - start_time
        if self.reporting:
            print("Backtest complete.")
            self.generate_reports()
            
    def close_all_positions(self):
        """
        Closes all open positions at the last available price.
        """
        # Generate EXIT signals for all symbols with open positions
        for symbol in self.symbols:
            position = self.portfolio.current_positions[symbol]
            if position.quantity != 0:
                # Determine the direction to exit the position
                if position.quantity > 0:
                    # Long position: need to sell
                    direction = 'SELL'
                else:
                    # Short position: need to buy
                    direction = 'BUY'

                # Get the latest price
                price = self.data_handler.get_latest_bar_value(symbol, 'Close')
                timestamp = self.data_handler.get_latest_bar_value(symbol, 'timestamp')

                # Create a market order to close the position
                order_event = OrderEvent(
                    symbol=symbol,
                    timestamp=timestamp,
                    order_type='MARKET',
                    quantity=abs(position.quantity),
                    direction=direction,
                    price=price
                )
                self.events.put(order_event)

        # Process the remaining events
        while not self.events.empty():
            event = self.events.get(False)
            if event.type == 'ORDER':
                self.execution_handler.execute_order(event)
            elif event.type == 'FILL':
                self.portfolio.update_fill(event)

    def generate_reports(self):
        # Create log directory for the strategy
        log_dir = os.path.join('log', self.strategy_name)
        os.makedirs(log_dir, exist_ok=True)
        print(f'Strategy report saved here: {log_dir}\n')

        # Generate reports
        report = PerformanceReport(self.data_handler, self.portfolio, execution_time=self.execution_time)
        equity = report.equity
        
        #symbol_performance = report.generate_symbol_report()

        # Save performance report & trades
        report_file = os.path.join(log_dir, 'report.txt')
        #report.save_report(report_file)
        report.save_trades(log_dir)

        # Save portfolio equity curve plot
        if len(self.symbols) > 1:
            plot_file = os.path.join(log_dir, 'equity_curve.html')
            plot_equity_curve(equity, self.initial_capital, save_path=plot_file, show_results=True)
        
            # Print performance summaries
            #performance = report.generate_report() #pass
            #self.print_performance_summary(performance, symbol_performance)

        # Plot per-symbol strategy results & save trades to file
        for symbol in self.symbols:
            # Prepare data for plotting
            data = self.data_handler.symbol_data[symbol]
            # Ensure data index is datetime
            data.index = pd.to_datetime(data.index)

            # Get trades for the symbol
            trades = pd.DataFrame(self.portfolio.trades)
            if not trades.empty:
                trades = trades[trades['symbol'] == symbol]

            # Create directories if not exist
            log_dir = os.path.join('log', self.strategy_name)
            os.makedirs(log_dir, exist_ok=True)

            # Plot strategy results
            report_file = os.path.join(log_dir, f'{symbol.replace('/', '')}_report.txt')
            symbol_performance = report.generate_report(symbol)
            report.save_report(report_file, symbol_performance)
            fig = plot_performance(symbol, data, trades, symbol_performance) 
            
            plot_file = os.path.join(log_dir, f'{symbol.replace('/', '')}_performance.html')
            fig.write_html(plot_file)
            self.logger.info(f"Performance plots were saved to file: {plot_file}")
            if self.open_in_browser:
                show_plotly_figure(fig) # WSL workaround, instead of fig.show()
            
            # Get indicators from strategy
            indicators = {}
            plot_locations = {}
            if symbol in self.strategy.indicator_data:
                symbol_indicators = self.strategy.indicator_data[symbol]
                timestamps = symbol_indicators.get('timestamps', [])
                for indicator_name, indicator_info in symbol_indicators.items():
                    if indicator_name == 'timestamps':
                        continue
                    values = indicator_info.get('values', [])
                    plot_location = indicator_info.get('plot_location', PlotLocation.MAIN)
                    indicators[indicator_name] = pd.Series(data=values, index=timestamps)
                    plot_locations[indicator_name] = plot_location.value  # Convert Enum to string
            
            if len(indicators) > 0:
                fig = plot_indicators(symbol, data, trades, indicators, plot_locations)
                plot_file = os.path.join(log_dir, f'{symbol.replace('/', '')}_indicators.html')
                fig.write_html(plot_file)
                self.logger.info(f"Indicators plots were saved to file: {plot_file}")
                #if self.show:
                #    show_plotly_figure(fig) # WSL workaround, instead of fig.show()
            
    def print_performance_summary(self, performance, symbol_performance):
        # Print performance summaries with limited decimal places
        print("Overall Performance Summary:")
        for key, value in performance.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

        if len(self.symbols) > 1:
            print("\nPer-Symbol Performance Summary:")
            for symbol, metrics in symbol_performance.items():
                print(f"\nSymbol: {symbol}")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
