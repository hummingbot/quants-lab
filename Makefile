.ONESHELL:
.PHONY: uninstall
.PHONY: install


uninstall:
	conda env remove -n quants-lab -y

install:
	./install.sh   # Run the full installation script
# Build local image
build:
	docker build -t hummingbot/quants-lab -f Dockerfile .
# Run db containers
run-db:
	docker compose -f docker-compose-db.yml up -d

# Stop db containers
stop-db:
	docker compose -f docker-compose-db.yml down

# Task Management Commands
# 
# Examples:
#   make run-tasks config=pools_screener_v2.yml              # Run tasks continuously (Docker)
#   make run-tasks config=pools_screener_v2.yml source=1     # Run tasks continuously (Local)
#   make trigger-task task=pools_screener config=pools_screener_v2.yml  # Run single task (Docker)
#   make serve-api config=pools_screener_v2.yml port=8000    # Start API server (Docker)
#   make list-tasks config=pools_screener_v2.yml             # List available tasks (Docker)

# Run tasks continuously
run-tasks:
ifeq ($(source),1)
	python cli.py run-tasks --config config/$(config)
else
	docker run -d --rm \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		--network host \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py run-tasks --config config/$(config)
endif

# Trigger single task
trigger-task:
ifeq ($(source),1)
	python cli.py trigger-task --task $(task) --config config/$(config)
else
	docker run --rm \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py trigger-task --task $(task) --config config/$(config)
endif

# Start API server with background tasks
serve-api:
ifeq ($(source),1)
	python cli.py serve --config config/$(config) --port $(port)
else
	docker run --rm \
		-p $(port):$(port) \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py serve --config config/$(config) --port $(port)
endif

# List available tasks
list-tasks:
ifeq ($(source),1)
	python cli.py list-tasks --config config/$(config)
else
	docker run --rm \
		-v $(shell pwd)/config:/quants-lab/config \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py list-tasks --config config/$(config)
endif

# Validate configuration file
validate-config:
ifeq ($(source),1)
	python cli.py validate-config --config config/$(config)
else
	docker run --rm \
		-v $(shell pwd)/config:/quants-lab/config \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py validate-config --config config/$(config)
endif

# Run task with Docker
run-task:
	docker run --rm \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		--network host \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py run-tasks --config config/$(config)

# Stop task runner (Docker)
stop-task:
	docker stop $(shell docker ps -q --filter ancestor=hummingbot/quants-lab) || true

# Launch Optuna Dashboard
launch-optuna:
	python -c "from core.backtesting.optimizer import StrategyOptimizer; optimizer = StrategyOptimizer(); optimizer.launch_optuna_dashboard()"

# Kill Optuna Dashboard
kill-optuna:
	python -c "from core.backtesting.optimizer import StrategyOptimizer; optimizer = StrategyOptimizer(); optimizer.kill_optuna_dashboard()"

# Clean up stale task states in MongoDB
cleanup-tasks:
	python scripts/cleanup_tasks.py

# List current task states
list-task-states:
	python scripts/cleanup_tasks.py --list

# Help target
help:
	@echo "QuantsLab Task Management Commands:"
	@echo ""
	@echo "🚀 Quick Start (Docker by default):"
	@echo "  make run-db                                     Start database (required first)"
	@echo "  make run-tasks config=CONFIG.yml               Run tasks continuously"
	@echo "  make run-notebook                               Run notebook task"
	@echo "  make stop-task                                  Stop running Docker tasks"
	@echo ""
	@echo "💻 Local Development (add source=1):"
	@echo "  make run-tasks config=CONFIG.yml source=1      Run tasks locally"
	@echo "  make run-notebook source=1                      Run notebook task locally"
	@echo ""
	@echo "📋 Task Commands (Docker by default):"
	@echo "  make run-tasks config=tasks/CONFIG.yml         Run tasks continuously"
	@echo "  make trigger-task task=NAME config=tasks/CONFIG.yml  Run single task"
	@echo "  make serve-api config=tasks/CONFIG.yml port=8000     Start API server"
	@echo "  make list-tasks config=tasks/CONFIG.yml        List available tasks"
	@echo "  make validate-config config=tasks/CONFIG.yml   Validate config"
	@echo ""
	@echo "🗄️ Database Commands:"
	@echo "  make run-db                                    Start database containers"
	@echo "  make stop-db                                   Stop database containers"
	@echo "  make cleanup-tasks                             Clean up stale task states"
	@echo "  make list-task-states                          List current task states"
	@echo ""
	@echo "📊 Optimization Commands:"
	@echo "  make launch-optuna                             Launch Optuna dashboard"
	@echo "  make kill-optuna                               Kill Optuna dashboard"
	@echo ""
	@echo "🔨 Build Commands:"
	@echo "  make build                                     Build Docker image"
	@echo "  make install                                   Run the installation script (install.sh)"
	@echo "  make uninstall                                 Remove conda environment"
	@echo ""
	@echo "📚 Examples:"
	@echo "  make run-tasks config=notebook_tasks.yml       # Docker"
	@echo "  make run-tasks config=notebook_tasks.yml source=1  # Local"
	@echo "  make run-notebook                               # Docker"
	@echo "  make run-notebook source=1                      # Local"
	@echo "  make trigger-task task=etl_download_candles config=notebook_tasks.yml"
	@echo ""
	@echo "📁 Directory Structure:"
	@echo "  app/outputs/        # All task outputs (notebooks, reports, etc.)"
	@echo "  app/data/           # Application data storage"
	@echo "  config/             # Task configuration files"
	@echo "  research_notebooks/ # Jupyter research notebooks"
