.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: type-hint
type-hint: upgradepip
	pytype facilyst -x facilyst/tests

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
installdeps: upgradepip
	pip install -e .

.PHONY: installdeps-test
installdeps-test: upgradepip
	pip install -e ".[test]"

.PHONY: installdeps-dev
installdeps-dev: upgradepip
	pip install -e ".[dev]"

.PHONY: test
test:
	pytest facilyst/tests -n 2 --durations 20 --timeout 300 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-datasets
test-datasets:
	pytest facilyst/tests/dataset_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-graphs
test-graphs:
	pytest facilyst/tests/graphs_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-mocks
test-mocks:
	pytest facilyst/tests/mock_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-models
test-models:
	pytest facilyst/tests/models_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-preprocessors
test-preprocessors:
	pytest facilyst/tests/preprocessing_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: test-utils
test-utils:
	pytest facilyst/tests//utils_tests -n 2 --durations 0 --cov=facilyst --junitxml=test-reports/git-all-tests-junit.xml

.PHONY: package_facilyst
package_facilyst: upgradepip
	python -m pip install --upgrade build
	python -m build
	$(eval DT_VERSION := $(shell grep '__version__\s=' facilyst/version.py | grep -o '[^ ]*$$'))
	tar -zxvf "dist/facilyst-${DT_VERSION}.tar.gz"
	mv "facilyst-${DT_VERSION}" unpacked_sdist