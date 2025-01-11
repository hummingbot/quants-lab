.ONESHELL:
.PHONY: uninstall
.PHONY: install
.PHONY: docker


uninstall:
	conda env remove -n quants-lab

install:
	conda env create -f environment.yml

docker:
	docker-compose up --build -d