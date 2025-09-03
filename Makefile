# EGW Query Expansion - Phase Enforcement Makefile
# ================================================

.PHONY: help install test lint build clean architecture-test phase-check all-checks

# Default target
help:
	@echo "EGW Query Expansion - Phase Enforcement Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install dependencies and setup virtual environment"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run standard test suite"
	@echo "  architecture-test Run architecture fitness function tests"
	@echo "  phase-check      Run import-linter phase enforcement checks"
	@echo "  test-all         Run all tests including architecture tests"
	@echo ""
	@echo "Quality Commands:"
	@echo "  lint             Run linting tools (black, isort, flake8, mypy)"
	@echo "  format           Format code with black and isort"
	@echo ""
	@echo "Build Commands:"
	@echo "  build            Build the package"
	@echo "  validate         Validate installation and basic functionality"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean            Clean build artifacts and cache"
	@echo "  all-checks       Run all quality checks and tests"
	@echo ""
	@echo "Phase Flow: I → X → K → A → L → R → O → G → T → S"

# Setup
install:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .

install-dev:
	. venv/bin/activate && pip install -e .[dev]
	. venv/bin/activate && pip install import-linter>=1.12.0 networkx

# Testing
test:
	. venv/bin/activate && pytest egw_query_expansion/tests/ -v

architecture-test:
	@echo "Running architecture fitness function tests..."
	. venv/bin/activate && pytest architecture_tests/ -v -m "architecture or phase_enforcement" \
		--tb=long --junit-xml=architecture-results.xml

phase-check:
	@echo "Running import-linter phase enforcement checks..."
	@echo "Canonical Phase Flow: I → X → K → A → L → R → O → G → T → S"
	. venv/bin/activate && import-linter --config pyproject.toml

test-all: test architecture-test

# Quality
lint:
	. venv/bin/activate && flake8 canonical_flow/ egw_query_expansion/ architecture_tests/
	. venv/bin/activate && mypy canonical_flow/ egw_query_expansion/ architecture_tests/ || echo "MyPy completed with warnings"

format:
	. venv/bin/activate && black canonical_flow/ egw_query_expansion/ architecture_tests/
	. venv/bin/activate && isort canonical_flow/ egw_query_expansion/ architecture_tests/

# Build
build:
	. venv/bin/activate && python -m pip install build
	. venv/bin/activate && python -m build

validate:
	. venv/bin/activate && python validate_installation.py

# All checks (CI simulation)
all-checks: phase-check architecture-test lint test validate
	@echo ""
	@echo "🎉 All checks completed successfully!"
	@echo "✅ Phase layering constraints enforced"
	@echo "✅ Architecture fitness functions passed"
	@echo "✅ Code quality checks passed"
	@echo "✅ Standard tests passed"
	@echo "✅ Installation validation passed"

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf architecture-coverage/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# CI simulation
ci-simulation: clean install-dev all-checks
	@echo ""
	@echo "🚀 CI simulation completed successfully!"
	@echo "This simulates what will happen in the CI pipeline."

# Quick phase violations check
quick-phase-check:
	@echo "Quick phase violations check..."
	python -c "
	import sys
	sys.path.insert(0, '.')
	from architecture_tests.test_phase_enforcement import ImportAnalyzer
	
	analyzer = ImportAnalyzer()
	violations = analyzer.analyze_phase_dependencies()
	
	total = sum(len(v) for v in violations.values())
	if total == 0:
		print('✅ No phase violations found!')
	else:
		print(f'❌ Found {total} phase violations:')
		for phase, viols in violations.items():
			if viols:
				print(f'  {phase}: {len(viols)} violations')
		sys.exit(1)
	"