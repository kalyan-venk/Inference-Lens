.PHONY: install setup lint test clean

install:
	pip install -e ".[dev]"

setup:
	mkdir -p data/raw data/processed reports/figures mlruns

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -delete
