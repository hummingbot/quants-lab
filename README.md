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

## Data Sources
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
