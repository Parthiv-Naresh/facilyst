import pandas as pd
import pytest

from facilyst.models import ADABoostRegressor, ModelBase
from facilyst.models.optimizers.hyperopt import HyperoptOptimizer
from facilyst.models.utils import get_models
from facilyst.utils import get_dataset


def test_hyperopt_class_variables():
    assert HyperoptOptimizer.name == "Hyperopt Optimizer"


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
    x, y = get_dataset("Sensor_Node_ALE")

    expected_iterations = {
        "ADA Boost Regressor": 20,
        "Bagging Regressor": 20,
        "Catboost Regressor": 20,
        "Decision Tree Regressor": 20,
        "Extra Trees Regressor": 20,
        "Random Forest Regressor": 20,
        "XGBoost Regressor": 20,
    }

    opt = HyperoptOptimizer(
        models=get_models("regression", exclude="neural"),
        iterations_per_model={
            "Random Forest Regressor": 20,
            ADABoostRegressor: 20,
            "xgboost regressor": 20,
            "decision tree regressor": 20,
            "Catboost Regressor": 20,
            "Extra Trees Regressor": 20,
            "Bagging Regressor": 20,
        },
    )
    best_model, best_score = opt.optimize(x, y)

    assert opt.iterations_per_model == expected_iterations
    assert isinstance(best_model, ModelBase)
    assert isinstance(best_score, float)
    assert isinstance(opt.results, pd.DataFrame)
    assert list(opt.results.columns) == ["best_hyperparameters", "best_score"]
    assert set(opt.results.index) == set(expected_iterations.keys())
    assert (opt.results.best_hyperparameters.map(type) == dict).all()
    best_model_hyperparameters = opt.results.loc[best_model.name].best_hyperparameters
    for hp_name, hp_val in best_model_hyperparameters.items():
        assert best_model.get_params()[hp_name] == best_model_hyperparameters[hp_name]
    print(opt.results)
