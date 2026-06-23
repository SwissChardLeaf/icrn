.PHONY: all install test test-unit test-notebooks lint coverage clean

# Do everything: lint, then run the full test suite.
all: lint test

# Run the full test suite: unit tests + tutorial notebooks.
test: test-unit test-notebooks

# Unit tests (icrn/test_*.py and icrn/**/test_*.py) with coverage.
test-unit:
	JAX_PLATFORMS=cpu coverage run -m unittest discover -s icrn -t . -v
	coverage report -m

# Execute every tutorial notebook and fail on any cell error.
test-notebooks:
	pytest --nbmake docs/tutorials/

# Run the pre-commit linters/formatters across the repo.
lint:
	pre-commit run --all-files

# Generate an HTML coverage report from the last `make test-unit` run.
coverage:
	coverage html
	@echo "open htmlcov/index.html"

clean:
	rm -rf .coverage coverage.xml htmlcov .pytest_cache
