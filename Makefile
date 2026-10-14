# Residualized Tone Space Makefile

.PHONY: help install test lint format clean build run docker-build docker-up docker-down

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	uv sync

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=residualized_tone_space --cov-report=html --cov-report=term

lint: ## Run linting
	uv run flake8 residualized_tone_space tests
	uv run mypy residualized_tone_space

format: ## Format code
	uv run black residualized_tone_space tests
	uv run isort residualized_tone_space tests

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

build: ## Build the package
	uv build

run: ## Run the API server
	uv run uvicorn residualized_tone_space.api.main:app --reload --host 0.0.0.0 --port 8000

run-jupyter: ## Run Jupyter Lab
	uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## Show Docker logs
	docker-compose logs -f

docker-shell: ## Open shell in Docker container
	docker-compose exec app bash

setup-env: ## Set up environment file
	cp .env.example .env
	@echo "Please edit .env with your configuration"

dev-setup: setup-env install ## Set up development environment
	@echo "Development environment set up successfully!"

ci: lint test ## Run CI pipeline (lint + test)

all: clean install test lint format ## Run all checks and formatting
