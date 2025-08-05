Your task is to develop a python based algorithmic trading event driven framework.

Inspired by Zorro [https://zorro-project.com](https://zorro-project.com/)

#### Core Requirements:
1. **Extensibility and Modularity:**
   - A plugin-based architecture to allow users to extend the framework with custom indicators, data feeds, and execution models.
   - Modular design to enable easy addition of new asset classes or order types.
2. **Event Processing:**
   - A robust event-driven architecture to handle real-time data processing, order management, and market events.

3. **Trading Capabilities:**
   - Support for both long and short trading strategies.
   - Implementation of both market and limit orders.

4. **Connectivity:**
   - Ability to connect to various data feeds, banks, brokers, or exchanges through connectors.

5. **Asset Support:**
   - Compatibility with multiple asset classes, including futures, stocks, ETFs, and cryptocurrencies.

6. **Backtesting and Live Trading:**
   - A backtest module that allows for seamless switching between backtesting and live trading using the same strategy.
   - Ability to include multiple assets in both backtesting and live trading for portfolio or pair trading strategies.

7. **Data Handling:**
   - Support for different historical data sources, including OHLC and tick data.
   - Capability to read data from CSV files and optionally use Parquet or custom binary file formats for improved performance.

7. **Strategy Optimization:**
   - Modules for strategy optimization using techniques like grid search or Bayesian optimization.
   - Support for Walk Forward Optimization (WFO) to provide out-of-sample backtesting, enhancing the robustness of the strategy.

8. **Monte Carlo Analysis:**
   - Implementation of Monte Carlo analysis to resample the equity curve from backtesting. 
   - Generation of multiple equity curves to represent different trade orders and price movements for performance analysis.

9. **Performance and Speed:**
   - Use of multiprocessing to utilize multiple CPU cores for faster backtesting.

10. **Charting and Reporting:**
   - Charting functionality using Plotly for visualizing source data and strategy results, such as equity curves.
   - Generation of comprehensive strategy performance reports, including metrics like the number of trades, winning percentage, win/loss ratio, Sharpe ratio, and maximum drawdown.

11. **Testing and Code Quality:**
    - Comprehensive code coverage with tests to ensure the reliability and stability of the framework.
