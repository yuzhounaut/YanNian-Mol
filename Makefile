# Makefile for lifespan predictor project

.PHONY: help install install-dev test test-unit test-integration test-all lint format type-check clean docs

help:
	@echo "Available commands:"
	@echo "  make install          - Install package and dependencies"
	@echo "  make install-dev      - Install package with development dependencies"
	@echo "  make test             - Run unit tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make test-all         - Run all tests including slow tests"
	@echo "  make lint             - Run linting checks (flake8)"
	@echo "  make format           - Format code with black and isort"
	@echo "  make type-check       - Run type checking with mypy"
	@echo "  make clean            - Clean build artifacts and cache"
	@echo "  make docs             - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v -m "not slow"

test-unit:
	pytest tests/ -v -m "unit and not slow"

test-integration:
	pytest tests/test_integration.py -v

test-all:
	pytest tests/ -v

lint:
	flake8 lifespan_predictor tests --count --statistics

format:
	black lifespan_predictor tests
	isort lifespan_predictor tests

type-check:
	mypy lifespan_predictor --config-file mypy.ini

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

docs:
	cd docs && make html
