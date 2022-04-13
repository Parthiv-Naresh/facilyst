import hyperopt.pyll.stochastic
import pandas as pd
import pytest
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from facilyst.models.optimizers.hyperopt import HyperoptOptimizer
from facilyst.models.utils import get_models
from facilyst.utils import get_dataset


def test_invalid_classifier_regressor_error():
    with pytest.raises(ValueError, match="Either classifier or regressor must be set."):
        HyperoptOptimizer(classifier=None, regressor=None)

    with pytest.raises(
        ValueError, match="Either classifier or regressor must be set, not both."
    ):
        HyperoptOptimizer(classifier="any", regressor="any")


def test_invalid_model_error():
    with pytest.raises(
        ValueError,
        match="The parameter `iterations_per_model` must be either an int or a dict",
    ):
        HyperoptOptimizer(regressor="any", iterations_per_model=50.0)


def test_hyperopt(numeric_features_regression):
    x = pd.DataFrame({"Col_1": [i for i in range(100)]})
    y = pd.Series([i for i in range(100)])

    opt = HyperoptOptimizer(
        regressor="Random Forest Regressor",
        iterations_per_model={"Random Forest Regressor": 5},
    )
    best_model, best_score = opt.optimize(x, y)

    expected_model = get_models("Random Forest Regressor")[0]

    assert best_model == expected_model(**best_model.get_params())
    assert isinstance(best_score, float)
