.ONESHELL:
.PHONY: uninstall
.PHONY: install


uninstall:
	conda env remove -n quants-lab -y

install:
	conda env create -f environment.yml
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
#   make run-tasks config=tasks/pools_screener_v2.yml              # Run tasks continuously
#   make trigger-task task=pools_screener config=tasks/pools_screener_v2.yml  # Run single task
#   make serve-api config=tasks/pools_screener_v2.yml port=8000    # Start API server
#   make list-tasks config=tasks/pools_screener_v2.yml             # List available tasks

# Run tasks continuously
run-tasks:
	python cli.py run-tasks --config config/$(config)

# Trigger single task
trigger-task:
	python cli.py trigger-task --task $(task) --config config/$(config)

# Start API server with background tasks
serve-api:
	python cli.py serve --config config/$(config) --port $(port)

# List available tasks
list-tasks:
	python cli.py list-tasks --config config/$(config)

# Validate configuration file
validate-config:
	python cli.py validate-config --config config/$(config)

# Run task with Docker
run-task:
	docker run --rm \
		-v $(shell pwd)/outputs:/quants-lab/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py run-tasks --config config/tasks/$(config)

# Run a specific notebook with Docker
run-notebook:
	docker run --rm \
		-v $(shell pwd)/outputs:/quants-lab/outputs \
		-v $(shell pwd)/config:/quants-lab/config \
		-v $(shell pwd)/app:/quants-lab/app \
		-v $(shell pwd)/research_notebooks:/quants-lab/research_notebooks \
		--env-file .env \
		hummingbot/quants-lab \
		conda run --no-capture-output -n quants-lab python3 cli.py run app.tasks.notebook.notebook_task

# Run task from source (local)
run-task-local:
	python cli.py run-tasks --config config/tasks/$(config)

# Run specific notebook from source (local)  
run-notebook-local:
	python cli.py run app.tasks.notebook.notebook_task

# Stop task runner (Docker)
stop-task:
	docker stop $(shell docker ps -q --filter ancestor=hummingbot/quants-lab) || true

# Help target
help:
	@echo "QuantsLab Task Management Commands:"
	@echo ""
	@echo "🚀 Quick Start (Docker):"
	@echo "  make run-task config=notebook_tasks.yml        Run task config in Docker"
	@echo "  make run-notebook                               Run notebook task directly"
	@echo "  make stop-task                                  Stop running Docker tasks"
	@echo ""
	@echo "💻 Local Development:"
	@echo "  make run-task-local config=notebook_tasks.yml  Run task config locally"
	@echo "  make run-notebook-local                         Run notebook task locally"
	@echo ""
	@echo "📋 Task Commands:"
	@echo "  make run-tasks config=tasks/CONFIG.yml         Run tasks continuously"
	@echo "  make trigger-task task=NAME config=tasks/CONFIG.yml  Run single task"
	@echo "  make serve-api config=tasks/CONFIG.yml port=8000     Start API server"
	@echo "  make list-tasks config=tasks/CONFIG.yml        List available tasks"
	@echo "  make validate-config config=tasks/CONFIG.yml   Validate config"
	@echo ""
	@echo "🗄️ Database Commands:"
	@echo "  make run-db                                    Start database containers"
	@echo "  make stop-db                                   Stop database containers"
	@echo ""
	@echo "🔨 Build Commands:"
	@echo "  make build                                     Build Docker image"
	@echo "  make install                                   Install conda environment"
	@echo "  make uninstall                                 Remove conda environment"
	@echo ""
	@echo "📚 Examples:"
	@echo "  make run-task config=notebook_tasks.yml"
	@echo "  make run-task config=data_collection_v2.yml"
	@echo "  make run-notebook"
	@echo "  make trigger-task task=etl_download_candles config=notebook_tasks.yml"
