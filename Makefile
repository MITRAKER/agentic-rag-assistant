# Short names for the commands in CONTRIBUTING.md. Run `make` to list them.
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install lint format typecheck power-of-ten test security check build clean

help:  ## List the commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-13s %s\n", $$1, $$2}'

install:  ## Install the project, its dev tools and the git hooks
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:  ## Lint with ruff
	ruff check .

format:  ## Format the code and fix what ruff can fix automatically
	ruff format .
	ruff check --fix .

typecheck:  ## Type-check with mypy in strict mode
	mypy

power-of-ten:  ## Check the NASA/JPL Power of Ten rules
	$(PYTHON) tools/power_of_ten.py src tools

test:  ## Run the tests with branch coverage (fails below 90%)
	pytest --cov --cov-report=term-missing

security:  ## Scan the code (bandit) and the dependencies (pip-audit)
	bandit -c pyproject.toml -r src tools
	pip-audit --skip-editable

check:  ## Run every check that CI runs (except the package build)
	pre-commit run --all-files
	$(MAKE) test security

build:  ## Build the wheel and source distribution into dist/
	$(PYTHON) -m build

clean:  ## Delete build output and tool caches
	rm -rf dist build .coverage coverage.xml htmlcov .pytest_cache .ruff_cache .mypy_cache
