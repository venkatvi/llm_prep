# Makefile for ML Framework
# Provides convenient commands for testing, formatting, and development

.PHONY: help install test test-autograd test-quick clean format lint setup-pre-commit

# Default target
help:
	@echo "Available commands:"
	@echo "  install          - Install all dependencies"
	@echo "  test             - Run comprehensive test suite"
	@echo "  test-autograd    - Run autograd tests only"
	@echo "  test-quick       - Run quick import tests"
	@echo "  format           - Format code with black and isort"
	@echo "  lint             - Lint code with flake8"
	@echo "  setup-pre-commit - Install pre-commit hooks"
	@echo "  clean            - Clean up temporary files"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8 pre-commit

# Run comprehensive test suite
test:
	@echo "Running comprehensive test suite..."
	python run_all_tests.py

# Run autograd tests only
test-autograd:
	@echo "Running autograd tests..."
	python run_all_tests.py --module autograd

# Run quick import tests
test-quick:
	@echo "Running quick tests..."
	python run_all_tests.py --module imports

# Format code
format:
	@echo "Formatting code..."
	black . --line-length 100
	isort . --profile black --line-length 100

# Lint code
lint:
	@echo "Linting code..."
	flake8 . --max-line-length=100 --extend-ignore=E203,W503
	@echo "Checking formatting..."
	black --check . --line-length 100 || echo "Run 'make format' to fix formatting"

# Setup pre-commit hooks
setup-pre-commit:
	@echo "Setting up pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed. Tests will run automatically before commits."

# Clean up temporary files
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true

# Development setup (install + pre-commit)
setup: install setup-pre-commit
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works."

# CI simulation (what runs in GitHub Actions)
ci:
	@echo "Simulating CI pipeline..."
	make clean
	make test-quick
	make test-autograd
	make lint
	@echo "CI simulation complete!"