PYTHON ?= python3
POETRY ?= poetry

.PHONY: bootstrap lint test format typecheck traces-demo docs

bootstrap:
	$(POETRY) install

lint:
	$(POETRY) run ruff check src tests

format:
	$(POETRY) run ruff format src tests

typecheck:
	$(POETRY) run mypy src

test:
	PYTHONPATH=src $(POETRY) run pytest

traces-demo:
	PYTHONPATH=src $(POETRY) run python examples/run_cli_example.py --demo-trace

ci:
	make lint typecheck test

docs:
	$(POETRY) run python scripts/render_docs.py
