# QuantsLab 🚀

QuantsLab is a comprehensive Python framework for quantitative research with Hummingbot. It provides a complete toolkit for data collection, backtesting, strategy development, and automated trading system deployment.

## Quick Start

### 🔥 One-Command Installation

```bash
# Clone the repository
git clone https://github.com/hummingbot/quants-lab.git
cd quants-lab

# Run the automated installer
./install.sh
```

The installer will:
- ✅ Check prerequisites (conda, docker, docker-compose)
- ✅ Create conda environment from `environment.yml`
- ✅ Install QuantsLab package in development mode
- ✅ Setup database (MongoDB only - TimescaleDB removed)
- ✅ Create `.env` file with defaults
- ✅ Test the complete installation

### Manual Installation

If you prefer manual setup:

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate quants-lab

# 2. Install package in development mode
pip install -e .

# 3. Start databases (optional)
docker compose -f docker-compose-db.yml up -d

# 4. Test installation
python cli.py --help
```

## Usage

### 📊 Command Line Interface

```bash
# List available tasks
python cli.py list-tasks

# Run tasks continuously from config
python cli.py run-tasks --config config/data_collection_v2.yml

# Run single task
python cli.py trigger-task --task pools_screener --config config/pools_screener_v2.yml

# Run task directly (no config needed!)
python cli.py run app.tasks.data_collection.pools_screener

# Start API server with background tasks
python cli.py serve --config config/data_collection_v2.yml --port 8000

# Validate configuration
python cli.py validate-config --config config/pools_screener_v2.yml
```

### 🔬 Research & Development

```bash
# Start Jupyter Lab for research
jupyter lab

# Research notebooks are located in:
# app/research_notebooks/
#   ├── ai_livestream/          # AI-powered trading strategies
#   ├── backtesting_examples/   # Strategy backtesting examples
#   ├── market_data_etl/   # Data exploration and analysis
#   ├── optimization_analysis/  # Parameter optimization studies
#   └── strategy_specific/      # Individual strategy research
```

### 💾 Database Access

**MongoDB UI**: http://localhost:28081/
- Username: `admin`
- Password: `changeme`

**Direct Connections**:
- MongoDB: `mongodb://admin:admin@localhost:27017/quants_lab`
- TimescaleDB: Removed - now using parquet files for time-series data

**Configuration**: All database settings are in `.env` file

## Architecture

```
quants-lab/
├── 🏗️  core/                    # Core framework (reusable library)
│   ├── backtesting/            # Backtesting engine with optimization
│   ├── data_sources/           # Market data integrations
│   ├── data_structures/        # Common data models
│   ├── features/               # Feature engineering & signals
│   └── tasks/                  # Task orchestration system
├── 📱 app/                     # Application layer (implementations)
│   ├── tasks/                  # Concrete task implementations
│   ├── controllers/            # Trading strategy controllers
│   ├── research_notebooks/     # Jupyter notebooks for research
│   └── data/                   # Application-specific data
├── ⚙️  config/                 # Configuration files (tasks & settings)
└── 🚀 cli.py                  # Command-line interface
```

## Key Features

### 📈 Data Sources & Storage
- **CLOB**: Order books, trades, candles, funding rates
- **AMM**: Liquidity data, pool stats, DEX analytics  
- **GeckoTerminal**: Multi-network DEX data, OHLCV
- **CoinGecko**: Market data, token & exchange stats
- **Parquet Storage**: High-performance time-series data storage with automatic caching

### 🧠 Research Tools
- **Triple Barrier Labeling**: Advanced ML labeling technique
- **Backtesting Engine**: Comprehensive strategy testing
- **Optimization**: Hyperparameter tuning with Optuna
- **Visualization**: Interactive charts and reports

### ⚡ Task System
- **Cron Scheduling**: Automated data collection
- **Dependency Management**: Task orchestration
- **Error Handling**: Robust failure recovery
- **API Integration**: RESTful task management

### 💾 Data Architecture
- **Parquet Files**: All time-series data (candles, trades) stored as parquet for optimal analytics performance
- **MongoDB**: Metadata, configuration, and application data
- **CLOB Integration**: Seamless caching and data management through CLOBDataSource
- **Automatic Merging**: Smart cache updates with incremental data fetching

## Quick Examples

### Data Collection
```python
# Collect pool data from multiple DEXs
python cli.py run app.tasks.data_collection.pools_screener --timeout 60

# Download historical candles
python cli.py run app.tasks.data_collection.candles_downloader_task
```

### Strategy Development
```python
# Import the framework
from core.backtesting.engine import BacktestEngine
from app.controllers.directional_trading.macd_bb_v1 import MACDBBV1Controller

# Run backtest
engine = BacktestEngine()
controller = MACDBBV1Controller()
results = engine.run(controller, start_date="2024-01-01")
```

### Research Workflow
1. **📊 Data Analysis**: Use notebooks in `app/research_notebooks/market_data_analysis/`
2. **🎯 Strategy Design**: Develop controllers in `app/controllers/`  
3. **🧪 Backtesting**: Test strategies with the backtesting engine
4. **⚡ Optimization**: Find best parameters using Optuna
5. **🚀 Deployment**: Deploy via task system

## Development

### Project Structure
- **Clean Architecture**: Separation between core framework and applications
- **Modular Design**: Easily extensible components
- **Type Safety**: Pydantic models with validation
- **Modern Python**: Python 3.12+ with latest libraries

### Adding New Strategies
1. Create controller in `app/controllers/`
2. Add backtesting notebook in `app/research_notebooks/`
3. Configure task in `config/`
4. Test with CLI: `python cli.py run app.tasks.your_task`

### Contributing
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Format code
black --line-length 130 .
isort --profile black --line-length 130 .
```

## Uninstallation

To completely remove QuantsLab:

```bash
./uninstall.sh
```

The uninstaller will:
- 🛑 Stop all running services and containers
- 🗑️ Remove the conda environment
- 🧹 Clean up Docker images, volumes, and networks  
- 📁 Remove Python cache and build artifacts
- ⚙️ Optionally remove generated data and outputs

Your source code and configuration files are preserved.

## Troubleshooting

**Import Errors**: Make sure you've installed the package with `pip install -e .`

**Database Connection Issues**: Ensure Docker containers are running:
```bash
docker-compose -f docker-compose-db.yml up -d
docker ps  # Check container status
```

**Task Failures**: Check logs and validate configuration:
```bash
python cli.py validate-config --config config/your_config.yml
```

**Port Conflicts**: If MongoDB port 27017 is in use, update `docker-compose-db.yml`

## Support

- 📚 **Documentation**: Check `ARCHITECTURE_V2.md` for detailed system design
- 🐛 **Issues**: Report bugs via GitHub issues
- 💡 **Ideas**: Contribute new strategies and features
- 🔧 **Development**: See `CLAUDE.md` for development guidelines

---

**Happy Trading! 🚀📈**