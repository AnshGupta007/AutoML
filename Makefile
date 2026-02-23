.PHONY: help install install-dev test lint format clean build run-api run-dashboard run-all docker-build docker-up docker-down

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy

# Colors for terminal
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

help: ## Show this help message
	@echo "AutoMLPro - Automated Machine Learning Pipeline"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ── Setup ──────────────────────────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: install ## Install development dependencies
	$(PIP) install -r requirements-dev.txt
	pre-commit install
	@echo "$(GREEN)Dev installation complete!$(NC)"

setup: ## Full environment setup
	$(PYTHON) scripts/setup_environment.sh
	make install-dev
	cp .env.example .env
	@echo "$(GREEN)Environment setup complete!$(NC)"

# ── Code Quality ───────────────────────────────────────────────────
format: ## Format code with black and isort
	$(BLACK) src/ tests/ scripts/
	$(ISORT) src/ tests/ scripts/
	@echo "$(GREEN)Code formatted!$(NC)"

lint: ## Run linters
	$(FLAKE8) src/ tests/
	$(MYPY) src/
	bandit -r src/ -ll
	@echo "$(GREEN)Linting complete!$(NC)"

format-check: ## Check formatting without making changes
	$(BLACK) --check src/ tests/
	$(ISORT) --check-only src/ tests/

# ── Testing ────────────────────────────────────────────────────────
test: ## Run all tests
	$(PYTEST) tests/ -v --tb=short

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v --tb=short

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) tests/unit/ -v --tb=short --no-cov

test-coverage: ## Run tests with coverage report
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing

test-performance: ## Run performance benchmarks
	$(PYTEST) tests/performance/ -v --tb=short --no-cov

# ── Running Services ───────────────────────────────────────────────
run-api: ## Run the FastAPI server locally
	uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload --log-level info

run-dashboard: ## Run the Streamlit dashboard
	streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

run-mlflow: ## Run MLflow tracking server
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///data/mlflow.db --default-artifact-root ./models/registry

run-all: ## Run all services (requires tmux or similar)
	@echo "Starting all services..."
	make run-mlflow &
	make run-api &
	make run-dashboard

# ── Pipeline ───────────────────────────────────────────────────────
train: ## Train a model (usage: make train DATA=path/to/data.csv TARGET=target_column)
	$(PYTHON) -m src.main train --data-path $(DATA) --target-column $(TARGET)

predict: ## Run predictions (usage: make predict DATA=path/to/data.csv MODEL=model_name)
	$(PYTHON) -m src.main predict --data-path $(DATA) --model-name $(MODEL)

pipeline: ## Run full end-to-end pipeline
	$(PYTHON) -m src.main pipeline --data-path $(DATA) --target-column $(TARGET)

# ── Docker ─────────────────────────────────────────────────────────
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all Docker services
	docker-compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Dashboard: http://localhost:8501"
	@echo "MLflow: http://localhost:5000"

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

docker-clean: ## Remove all Docker resources
	docker-compose down -v --rmi all
	docker system prune -f

# ── Database ───────────────────────────────────────────────────────
db-init: ## Initialize database
	$(PYTHON) -c "from src.utils.db import init_db; init_db()"

db-migrate: ## Run database migrations
	alembic upgrade head

# ── Reports ────────────────────────────────────────────────────────
generate-report: ## Generate evaluation report
	$(PYTHON) scripts/generate_report.py

# ── Cleanup ────────────────────────────────────────────────────────
clean: ## Remove build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "$(GREEN)Cleaned up!$(NC)"

clean-data: ## Remove processed data (keep raw)
	rm -rf data/processed/* data/features/* data/predictions/*

clean-models: ## Remove trained models
	rm -rf models/trained/* models/onnx/*

# ── Documentation ─────────────────────────────────────────────────
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

# ── Release ────────────────────────────────────────────────────────
build: ## Build distribution packages
	$(PYTHON) -m build

release: format lint test build ## Full release (format + lint + test + build)
	@echo "$(GREEN)Release ready!$(NC)"

# ── Quick Start ────────────────────────────────────────────────────
quickstart: ## Quick start demo with sample data
	@echo "$(YELLOW)Running AutoMLPro quickstart demo...$(NC)"
	$(PYTHON) -m src.main pipeline \
		--data-path tests/test_data/sample_classification.csv \
		--target-column target \
		--task-type classification \
		--experiment-name quickstart_demo
