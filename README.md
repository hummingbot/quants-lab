# QuantsLab

QuantsLab is a Python project designed for quantitative research with Hummingbot. It provides functionalities for fetching historical data, calculating metrics, backtesting, and generating trading configurations.

## Installation

### Prerequisites
- Anaconda (or Miniconda) must be installed on your system. You can download it from [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/hummingbot/quants-lab.git
   cd quants-lab
   ```

2. Create and activate the Conda environment:
   ```
   make install
   ```
   This command will create a new Conda environment and install all the necessary dependencies.

3. Activate the environment:
   ```
   conda activate quants-lab
   ```

You're now ready to use QuantsLab!

## Usage
1. Research Notebooks
    - Under the research notebooks folder you will find a folder for each strategy. The idea is that you can use them as an inspiration to do research on your own strategies.
    - The main steps are:
        - Exploratory Data Analysis
        - Design the controller
        - Backtest a simple controller
        - Optimize and find the best parameters
2. Tasks
   - Under the tasks folder you will find the task runner that is the entrypoint to run periodic tasks. There are some examples under data collection but basically the execute method is the one that will be called by the task runner at the specified intervals.
   - You can run it from source but you will probably want to do it from a docker container. You will need to build the environment first with the following command:
     ```
     docker build -t hummingbot/quants_lab .
     ```
     And then you can use the docker-compose file to run the container:
     ```
        docker-compose up
        ```
   - Is important to check if your task will require a database or not. The examples provided are using a database (that is also in the compose file) but you can also use the task runner without a database.

## Data Source
- **CLOB (Central Limit Order Book)**
  - Last Traded Price
  - Current Order Book
  - Historical Candles
  - Historical Trades
  - Trading Rules
  - Funding Info

- **AMM (Automated Market Maker)**
  - Last Traded Price
  - Current Liquidity
  - Pool Stats
    - Fees Collected
    - Volume (24h)
  - Historical Trades

- **GeckoTerminal**
  - Networks
  - Dexes by Network
  - Top Pools by Network
  - Top Pools by Network Dex
  - Top Pools by Network Token
  - New Pools by Network
  - New Pools (All Networks)
  - OHLCV

- **CoinGecko**
  - Top Tokens Stats
  - Top Exchange Stats
  - Market Stats by Token
  - Market Stats by Exchange

- **Spice (DuneAnalytics)**
  - Queries

### Modules
- **Labeling**
  - Triple Barrier Method

- **Backtesting**

- **Optimization**

- **Visualization**
  - OHLC
  - Order Book
  - Backtesting Report

- **Features**
  - Signals
