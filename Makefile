.PHONY: help install test lint format type-check analysis clean dev-setup pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install dependencies"
	@echo "  dev-setup      - Set up development environment"
	@echo "  lint           - Run ruff linting"
	@echo "  format         - Run code formatting (black, isort)"
	@echo "  type-check     - Run mypy type checking"
	@echo "  analysis       - Run full static analysis suite"
	@echo "  pre-commit     - Install and run pre-commit hooks"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean build artifacts"
	@echo "  ci-validate    - Run CI validation checks"

# Installation and setup
install:
	pip install -r requirements.txt

dev-setup: install
	pip install -e .
	pre-commit install
	mkdir -p analysis_reports

# Code quality checks
lint:
	ruff check --config pyproject.toml egw_query_expansion/
	@echo "✅ Linting complete"

format:
	black egw_query_expansion/
	ruff format egw_query_expansion/
	isort egw_query_expansion/
	@echo "✅ Code formatting complete"

type-check:
	mypy --config-file mypy.ini egw_query_expansion/
	@echo "✅ Type checking complete"

# Comprehensive static analysis
analysis:
	@echo "🚀 Running comprehensive static analysis..."
	python scripts/run_strict_analysis.py

# Static analysis components
check-star-imports:
	python scripts/check_star_imports.py $$(find egw_query_expansion -name "*.py")

check-circular-imports:
	python scripts/check_circular_imports.py $$(find egw_query_expansion -name "*.py")

check-type-imports:
	python scripts/validate_type_imports.py $$(find egw_query_expansion -name "*.py")

# Pre-commit integration
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Testing
test:
	pytest egw_query_expansion/tests/ -v

test-coverage:
	pytest --cov=egw_query_expansion --cov-report=html --cov-report=term egw_query_expansion/tests/

# CI validation (matches CI pipeline)
ci-validate:
	@echo "🔍 Running CI validation locally..."
	mypy --strict --config-file mypy.ini --show-error-codes egw_query_expansion/
	ruff check --config pyproject.toml egw_query_expansion/
	python scripts/check_star_imports.py $$(find egw_query_expansion -name "*.py")
	python scripts/check_circular_imports.py $$(find egw_query_expansion -name "*.py")
	python scripts/validate_type_imports.py $$(find egw_query_expansion -name "*.py")
	ruff check --select I --config pyproject.toml egw_query_expansion/
	black --check egw_query_expansion/
	@echo "✅ CI validation complete"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf analysis_reports/
	@echo "✅ Cleanup complete"

# Development workflow shortcuts
quick-check: lint type-check
	@echo "✅ Quick checks complete"

full-check: format lint type-check test
	@echo "✅ Full validation complete"

# Fix common issues automatically
fix:
	ruff check --fix --config pyproject.toml egw_query_expansion/
	black egw_query_expansion/
	isort egw_query_expansion/
	@echo "✅ Auto-fixes applied"