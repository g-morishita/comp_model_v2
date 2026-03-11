.PHONY: lint format typecheck test test-all check

lint:
	uv run ruff check src/ tests/ --fix

format:
	uv run ruff format src/ tests/

typecheck:
	uv run pyright src/

test:
	uv run pytest -m "not stan" --tb=short -q

test-all:
	uv run pytest --tb=short -q

check: lint format typecheck test

