import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def one_dim_data():
    one_dim = [i for i in range(10)]
    one_dim_types = {
        "pandas": pd.Series(one_dim),
        "numpy": np.array(one_dim),
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
