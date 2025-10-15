.ONESHELL:
.PHONY: help install uninstall build

# Default target
.DEFAULT_GOAL := help

# ============================================================================
# INSTALLATION & SETUP
# ============================================================================

install:  ## Install QuantsLab (conda env + package + databases)
	@echo "🚀 Installing QuantsLab..."
	./install.sh

uninstall:  ## Remove conda environment completely
	@echo "🗑️  Removing QuantsLab environment..."
	conda env remove -n quants-lab -y

build:  ## Build Docker image locally
	@echo "🔨 Building Docker image..."
	docker build -t hummingbot/quants-lab -f Dockerfile .
# ============================================================================
# DATABASE MANAGEMENT
# ============================================================================

run-db:  ## Start MongoDB and Mongo Express
	@echo "🗄️  Starting databases..."
	docker compose -f docker-compose-db.yml up -d
	@echo "✅ MongoDB: mongodb://admin:admin@localhost:27017/quants_lab"
	@echo "✅ Mongo Express UI: http://localhost:28081 (admin/changeme)"

stop-db:  ## Stop database containers
	@echo "🛑 Stopping databases..."
	docker compose -f docker-compose-db.yml down

logs-db:  ## View database logs
	docker compose -f docker-compose-db.yml logs -f

clean-db:  ## Stop databases and remove volumes (⚠️  DATA LOSS)
	@echo "⚠️  WARNING: This will delete all database data!"
	@read -p "Are you sure? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	docker compose -f docker-compose-db.yml down -v

# ============================================================================
# TASK MANAGEMENT
# ============================================================================
# Run tasks in Docker by default, add 'source=1' for local execution
# Examples:
#   make run-tasks config=tf_pipeline.yml              # Docker
#   make run-tasks config=tf_pipeline.yml source=1     # Local
#   make trigger-task task=data_collection config=tf_pipeline.yml
# ============================================================================

run-tasks:  ## Run tasks continuously (Docker or local with source=1)
ifeq ($(source),1)
	@echo "▶️  Running tasks locally: $(config)"
	python cli.py run-tasks --config config/$(config)
else
	@echo "🐳 Running tasks in Docker: $(config)"
	@docker run -d --rm \
		--name quants-lab-$(shell echo $(config) | sed 's/\.yml//') \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		--network host \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py run-tasks --config config/$(config)
	@echo "✅ Task runner started in background"
	@echo "   View logs: make logs-tasks"
	@echo "   Stop: make stop-tasks"
endif

trigger-task:  ## Run a single task once (Docker or local with source=1)
ifeq ($(source),1)
	@echo "⚡ Triggering task locally: $(task)"
	python cli.py trigger-task --task $(task) --config config/$(config)
else
	@echo "🐳 Triggering task in Docker: $(task)"
	docker run --rm \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		--network host \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py trigger-task --task $(task) --config config/$(config)
endif

serve-api:  ## Start API server with background tasks
ifeq ($(source),1)
	@echo "🌐 Starting API server locally on port $(port)"
	python cli.py serve --config config/$(config) --port $(port)
else
	@echo "🐳 Starting API server in Docker on port $(port)"
	docker run -d --rm \
		--name quants-lab-api \
		-p $(port):$(port) \
		-v $(shell pwd)/app/outputs:/quants-lab/app/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		--network host \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py serve --config config/$(config) --port $(port)
	@echo "✅ API server started at http://localhost:$(port)"
endif

list-tasks:  ## List all tasks from config file
ifeq ($(source),1)
	@python cli.py list-tasks --config config/$(config)
else
	@docker run --rm \
		-v $(shell pwd)/config:/quants-lab/config \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py list-tasks --config config/$(config)
endif

validate-config:  ## Validate task configuration file
ifeq ($(source),1)
	@python cli.py validate-config --config config/$(config)
else
	@docker run --rm \
		-v $(shell pwd)/config:/quants-lab/config \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py validate-config --config config/$(config)
endif

stop-tasks:  ## Stop all running task containers
	@echo "🛑 Stopping all task runners..."
	@docker ps --filter ancestor=hummingbot/quants-lab --format "{{.Names}}" | xargs -r docker stop || true
	@echo "✅ All task runners stopped"

logs-tasks:  ## View logs from running tasks
	@docker ps --filter ancestor=hummingbot/quants-lab --format "{{.Names}}" | head -1 | xargs -r docker logs -f

ps-tasks:  ## List running task containers
	@echo "📋 Running task containers:"
	@docker ps --filter ancestor=hummingbot/quants-lab --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}"

# ============================================================================
# OPTIMIZATION & UTILITIES
# ============================================================================

launch-optuna:  ## Launch Optuna dashboard for hyperparameter optimization
	@echo "📊 Launching Optuna dashboard..."
	python -c "from core.backtesting.optimizer import StrategyOptimizer; optimizer = StrategyOptimizer(); optimizer.launch_optuna_dashboard()"

kill-optuna:  ## Stop Optuna dashboard
	@echo "🛑 Stopping Optuna dashboard..."
	python -c "from core.backtesting.optimizer import StrategyOptimizer; optimizer = StrategyOptimizer(); optimizer.kill_optuna_dashboard()"

# ============================================================================
# MAINTENANCE & CLEANUP
# ============================================================================

cleanup-tasks:  ## Clean up stale task states in MongoDB
	@echo "🧹 Cleaning up stale tasks..."
	python scripts/cleanup_tasks.py

list-task-states:  ## List current task states in MongoDB
	@python scripts/cleanup_tasks.py --list

clean:  ## Remove Python cache and build artifacts
	@echo "🧹 Cleaning Python cache and build artifacts..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/ 2>/dev/null || true
	@echo "✅ Cleanup complete"

# ============================================================================
# HELP
# ============================================================================

help:  ## Show this help message
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  QuantsLab - Quantitative Trading Framework"
	@echo "════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "📦 INSTALLATION & SETUP"
	@grep -E '^install:|^uninstall:|^build:' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🗄️  DATABASE MANAGEMENT"
	@grep -E '^run-db:|^stop-db:|^logs-db:|^clean-db:' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "⚡ TASK MANAGEMENT (add 'source=1' for local execution)"
	@grep -E '^run-tasks:|^trigger-task:|^stop-tasks:|^logs-tasks:|^ps-tasks:' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📋 CONFIGURATION"
	@grep -E '^list-tasks:|^validate-config:' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "📊 OPTIMIZATION & API"
	@grep -E '^serve-api:|^launch-optuna:|^kill-optuna:' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "🧹 MAINTENANCE"
	@grep -E '^cleanup-tasks:|^list-task-states:|^clean:' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "════════════════════════════════════════════════════════════════"
	@echo "🚀 QUICK START"
	@echo "════════════════════════════════════════════════════════════════"
	@echo "  1. make install                    # Install everything"
	@echo "  2. conda activate quants-lab       # Activate environment"
	@echo "  3. make run-db                     # Start databases"
	@echo "  4. make run-tasks config=tf_pipeline.yml  # Run tasks"
	@echo ""
	@echo "📚 EXAMPLES"
	@echo "  make run-tasks config=tf_pipeline.yml          # Docker (background)"
	@echo "  make run-tasks config=tf_pipeline.yml source=1 # Local execution"
	@echo "  make trigger-task task=data_collection config=tf_pipeline.yml"
	@echo "  make logs-tasks                                # View task logs"
	@echo "  make stop-tasks                                # Stop all tasks"
	@echo ""
