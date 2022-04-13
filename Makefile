.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	isort --check-only facilyst --profile black
	black facilyst -t py39 --check

.PHONY: lint-fix
lint-fix:
	isort facilyst --profile black
	black -t py39 facilyst
	pydocstyle facilyst/ --convention=google --add-ignore=D104,D105,D107 --add-select=D400 --match-dir='^(?!(tests)).*'

.PHONY: installdeps
installdeps:
	pip install --upgrade pip -q
	pip install -e .

.PHONY: installdeps-test
installdeps-test:
	pip install -e . -q
	pip install -r test-requirements.txt

.PHONY: installdeps-dev
installdeps-dev:
	pip install -e . -q
	pip install -r dev-requirements.txt

.PHONY: test
test:
	pytest facilyst/tests -n 2 --durations 20 --timeout 300 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-datasets
test-datasets:
	pytest facilyst/tests/dataset_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-models
test-models:
	pytest facilyst/tests/models_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-mocks
test-mocks:
	pytest facilyst/tests//mock_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-graphs
test-graphs:
	pytest facilyst/tests//graphs_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-utils
test-utils:
	pytest facilyst/tests//utils_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml