.ONESHELL:
.PHONY: uninstall
.PHONY: install
.PHONY: run


uninstall:
	conda env remove -n quants-lab

install:
	conda env create -f environment.yml
