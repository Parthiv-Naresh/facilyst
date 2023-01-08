import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from facilyst.models import ModelBase, TimeSeriesModelBase
from facilyst.utils import make_wave


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_extra_dependency: marks test as needing [extra] dependencies"
    )


@pytest.fixture
def one_dim_data():
    one_dim = [i for i in range(10)]
    one_dim_types = {
        "pandas": pd.Series(one_dim),
        "numpy": np.array(one_dim, dtype=np.int64),
        "list": one_dim,
        "str": 0,
        "None": None,
    }

    return one_dim_types


@pytest.fixture
def multi_dim_data():
    multi_dim = [[i for i in range(10)] for j in range(10)]
    multi_dim_types = {
        "pandas": pd.DataFrame(multi_dim),
        "numpy": np.array(multi_dim),
        "list": multi_dim,
        "None": None,
    }

    return multi_dim_types


@pytest.fixture
def numeric_features_regression():
    X, y = make_regression(n_samples=100, n_features=10)
    return X, y


@pytest.fixture
def numeric_features_binary_classification():
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, n_informative=2
    )
    return X, y


@pytest.fixture
def numeric_features_multi_classification():
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=3, n_informative=3
    )
    return X, y


@pytest.fixture
def time_series_data():
    def _get_data(
        make_index_datetime_x=True,
        make_index_datetime_y=True,
        numeric_features=True,
        freq="D",
        target_wave=(1.0, 1.0, 0.0),
        num_rows=100,
    ):
        dateindex_ = pd.date_range("1/1/1956", periods=num_rows, freq=freq)
        rangeindex_ = np.array([i for i in range(num_rows)])
        if make_index_datetime_x:
            x_index = dateindex_
        else:
            x_index = rangeindex_
        if make_index_datetime_y:
            y_index = dateindex_
        else:
            y_index = rangeindex_

        x = pd.DataFrame(index=x_index)

        y = make_wave(
            num_rows=num_rows,
            library="numpy",
            amplitude=target_wave[0],
            frequency=target_wave[1],
            trend=target_wave[2],
        )

        if numeric_features:
            x["floats_0"] = (y * 1.723) ** 3
            x["floats_1"] = ((y + 1) ** 2) * 1.345

        y = pd.Series(y, index=y_index)

        train_size = int(num_rows * 0.8)
        x_train = x.iloc[:train_size]
        x_test = x.iloc[train_size:]
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        return x_train, x_test, y_train, y_test

    return _get_data


@pytest.fixture()
def mock_regression_model_class():
    class MockRegressionModel(ModelBase):
        name = "Mock Regression Model"
        primary_type = "regression"
        secondary_type = "ensemble"
        tertiary_type = "tree"
        hyperparameters = {}

        def __init__(self, first_arg=10):
            super().__init__(
                model=None,
                parameters={"first_arg": first_arg},
            )

        def fit(self, y_train, x_train=None):
            return self

        def get_params(self):
            return {}

    return MockRegressionModel


@pytest.fixture()
def mock_time_series_model_class():
    class MockTSModel(TimeSeriesModelBase):
        name = "Mock TS Model"
        primary_type = "regression"
        secondary_type = "time series"
        tertiary_type = "random"
        hyperparameters = {}

        def __init__(self, first_arg=10):
            super().__init__(
                model=None,
                parameters={"first_arg": first_arg},
            )

        def fit(self, y_train, x_train=None):
            return self

        def get_params(self):
            return {}

    return MockTSModel


def pytest_addoption(parser):
    parser.addoption(
        "--no-extra-dependencies",
        action="store_true",
        default=False,
        help="If true, tests will assume no [extra] dependencies have been installed.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-extra-dependencies"):
        skip_extra = pytest.mark.skip(reason="Needs an [extra] dependency")
        for item in items:
            if "needs_extra_dependency" in item.keywords:
                item.add_marker(skip_extra)


@pytest.fixture
def has_no_extra_dependencies(pytestconfig):
    return pytestconfig.getoption("--no-extra-dependencies")
