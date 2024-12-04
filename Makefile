# Makefile for managing Conda environment and Docker Compose services
.ONESHELL:
.PHONY: uninstall install start stop clean rebuild

# Conda environment management
ENV_NAME = quants-lab

uninstall:
	@echo "Removing Conda environment: $(ENV_NAME)"
	conda env remove -n $(ENV_NAME)

install:
	@echo "Creating Conda environment from environment.yml"
	conda env create -f environment.yml

# Docker Compose services
COMPOSE_DB = docker compose -f compose-db.yml
COMPOSE_TASKS = docker compose -f compose-tasks.yml

start:
	@if [ "$(SERVICE)" ]; then \
		$(COMPOSE_DB) up -d $(SERVICE) || $(COMPOSE_TASKS) up -d $(SERVICE); \
	else \
		echo "Please specify a SERVICE to start. Use 'make start SERVICE=<service_name>'"; \
	fi

stop:
	@if [ "$(SERVICE)" ]; then \
		$(COMPOSE_DB) down $(SERVICE) || $(COMPOSE_TASKS) down $(SERVICE); \
	else \
		echo "Please specify a SERVICE to stop. Use 'make stop SERVICE=<service_name>'"; \
	fi

# Convenience shortcuts for database services
start-timescaledb:
	$(COMPOSE_DB) up -d timescaledb

stop-timescaledb:
	$(COMPOSE_DB) down timescaledb

start-optunadb:
	$(COMPOSE_DB) up -d optunadb

stop-optunadb:
	$(COMPOSE_DB) down optunadb

# Convenience shortcuts for task runners
start-trades:
	$(COMPOSE_TASKS) up -d data-generation-runner

stop-trades:
	$(COMPOSE_TASKS) down data-generation-runner

start-candles:
	$(COMPOSE_TASKS) up -d candles-downloader-runner

stop-candles:
	$(COMPOSE_TASKS) down candles-downloader-runner

start-report:
	$(COMPOSE_TASKS) up -d screeners-report-runner

stop-report:
	$(COMPOSE_TASKS) down screeners-report-runner

# Clean and rebuild all containers
clean:
	$(COMPOSE_DB) down -v
	$(COMPOSE_TASKS) down -v

rebuild:
	$(COMPOSE_DB) down --rmi all -v
	$(COMPOSE_TASKS) down --rmi all -v
	$(COMPOSE_DB) up --build -d
	$(COMPOSE_TASKS) up --build -d

# Usage help
# Usage help
help:
	@echo "Available targets:"
	@echo "  install              - Create Conda environment from environment.yml"
	@echo "  uninstall            - Remove Conda environment"
	@echo "  start SERVICE=<name> - Start a specific service"
	@echo "  stop SERVICE=<name>  - Stop a specific service"
	@echo "  start-optunadb       - Start optuna database"
	@echo "  stop-optunadb        - Stop optuna database"
	@echo "  start-timescaledb    - Start timescale database"
	@echo "  stop-timescaledb     - Stop timescale database"
	@echo "  start-db             - Start database services (timescaledb, optunadb)"
	@echo "  stop-db              - Stop database services"
	@echo "  start-trades         - Start trades task runner"
	@echo "  stop-trades          - Stop trades task runner"
	@echo "  start-candles        - Start candles downloader"
	@echo "  stop-candles         - Stop candles downloader"
	@echo "  start-report         - Start report generator"
	@echo "  stop-report          - Stop report generator"
	@echo "  clean                - Remove all containers and volumes"
	@echo "  rebuild              - Rebuild and start all services"