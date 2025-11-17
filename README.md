# QuantsLab 🚀

Python framework for quantitative trading research with Hummingbot. Built for data collection, backtesting, strategy development, and automated deployment.

## Quick Start

### Installation

```bash
git clone https://github.com/hummingbot/quants-lab.git
cd quants-lab
make install
```

The installer sets up:
- Conda environment (Python 3.12)
- All dependencies
- MongoDB database
- Configuration files

### Deploy a Recurring Task

```bash
# 1. Activate environment
conda activate quants-lab

# 2. Start database
make run-db

# 3. Run tasks (Docker - recommended for production)
make run-tasks config=tf_pipeline.yml

# 4. View logs and monitor
make logs-tasks
make ps-tasks

# 5. Stop when done
make stop-tasks
```

**Local development mode:**
```bash
make run-tasks config=tf_pipeline.yml source=1
```

## Key Commands

Type `make` or `make help` to see all commands.

**Installation:**
- `make install` - Full installation
- `make build` - Build Docker image
- `make uninstall` - Remove environment

**Database:**
- `make run-db` - Start MongoDB
- `make stop-db` - Stop MongoDB
- Mongo Express UI: http://localhost:28081 (admin/changeme)

**Tasks:**
- `make run-tasks config=FILE.yml` - Run continuously (Docker)
- `make run-tasks config=FILE.yml source=1` - Run locally
- `make trigger-task task=NAME config=FILE.yml` - Run once
- `make logs-tasks` - View logs
- `make stop-tasks` - Stop all tasks
- `make ps-tasks` - List running tasks

**Configuration:**
- `make list-tasks config=FILE.yml` - List available tasks
- `make validate-config config=FILE.yml` - Validate config

## Architecture

```
quants-lab/
├── core/                  # Reusable framework
│   ├── backtesting/       # Backtesting engine + optimizer
│   ├── data_sources/      # Market data integrations (CLOB, AMM, APIs)
│   ├── features/          # Feature engineering & signals
│   └── tasks/             # Task orchestration system
├── app/                   # Application layer
│   ├── tasks/             # Task implementations
│   └── data/              # Application data
├── controllers/           # Trading strategies
├── config/                # Task configurations (YAML)
├── research_notebooks/    # Jupyter notebooks
└── cli.py                # Command-line interface
```

## Configuration Files

Task configurations are YAML files in `config/`:

```yaml
tasks:
  data_collection:
    enabled: true
    task_class: app.tasks.notebook.notebook_task.NotebookTask
    schedule:
      type: interval
      hours: 6
    config:
      notebooks:
        - data_collection/download_candles_all_pairs.ipynb
        - feature_engineering/trend_follower_grid.ipynb
      output_dir: app/outputs/cohort-12
```

## Development

```bash
# Activate environment
conda activate quants-lab

# Run Jupyter for research
jupyter lab

# List available tasks
make list-tasks config=tf_pipeline.yml

# Format code
black --line-length 130 .
isort --profile black --line-length 130 .
```

## Database Access

- **MongoDB**: `mongodb://admin:admin@localhost:27017/quants_lab`
- **Mongo Express UI**: http://localhost:28081 (admin/changeme)
- **Config**: All settings in `.env` file

## Data Sources

- **CLOB**: Order books, trades, candles, funding rates
- **AMM**: DEX liquidity and pool data
- **GeckoTerminal**: Multi-network OHLCV data
- **CoinGecko**: Market data and stats

## Troubleshooting

**Database connection issues:**
```bash
make run-db
docker ps  # Verify containers running
```

**Task failures:**
```bash
make logs-tasks  # View logs
make validate-config config=YOUR_CONFIG.yml
```

**Port conflicts:**
Edit `docker-compose-db.yml` if port 27017 or 28081 are in use.

## Support

- 📚 Documentation: See `CLAUDE.md` for dev guidelines
- 🐛 Issues: GitHub issues
- 💡 Contributing: Fork and submit PRs

---

**Happy Trading! 🚀📈**
