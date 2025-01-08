.ONESHELL:
.PHONY: uninstall
.PHONY: install


uninstall:
	conda env remove -n quants-lab

install:
	conda env create -f environment.yml

# Run task runner with specified config
run-task:
	TASK_CONFIG=config/$(config) docker-compose -f docker-compose-task-runner.yml up task-runner

# Stop task runner
stop-task:
	docker-compose -f docker-compose-task-runner.yml down
