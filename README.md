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
### 1. Research Notebooks
   - Under the research notebooks folder you will find a folder for each strategy. The idea is that you can use them as an inspiration to do research on your own strategies.
   - The main steps are:
       - Exploratory Data Analysis
       - Design the controller
       - Backtest a simple controller
       - Optimize and find the best parameters

        
---     

### 2. Task Orchestration

#### **Prerequisites**
1. Ensure Docker and Docker Compose are installed. If not, you can install them with the following commands:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    ```

2. Verify the Docker installation by running:
    ```bash
    docker --version
    docker compose version
    ```

---

#### **Configuration**

1. **Modify Task Settings**:
   - Navigate to the `config` folder and locate the `tasks.yml` file.
   - Update the file to include the specific tasks you want to execute. By default, it is configured to run the pool-fetching task.

2. **Customize Database Credentials** *(Optional)*:
   - The default credentials for MongoDB and PostgreSQL are specified in the `docker-compose-db.yml` file.
   - Update these credentials if necessary, especially for production environments.

---

#### **Steps to Run Tasks**

1. **Build the Docker Image**:
   Build the local Quants-Lab Docker image by running:
   ```bash
   make build
   ```

2. **Start Databases**:
   Start the necessary databases (MongoDB and PostgreSQL) using:
   ```bash
   make run-db
   ```

3. **Run the Task Runner**:
   Execute tasks specified in `tasks.yml` with the following command:
   ```bash
   make run-task config=tasks.yml
   ```

4. **Monitor Database Activity**:
   Use Mongo Compass UI to inspect the database data:
   - Open your web browser and visit:
     ```
     http://localhost:28081/
     ```
   - Default credentials:
     - **Username**: `admin`
     - **Password**: `changeme`
   - Update these credentials in `docker-compose-db.yml` if needed.

   Replace `localhost` with your machine's IP address if accessing remotely.

---

#### **Stopping Services**

1. **Stop the Task Runner**:
   ```bash
   make stop-task
   ```

2. **Stop the Databases**:
   ```bash
   make stop-db
   ```

---

### Notes:
- Ensure all required ports are open and accessible.
- Regularly check the logs for errors using `docker logs <container_name>`.
- Make sure to re-build the local Docker image using `make build` after any changes are made to `tasks.yml`.

--- 

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
