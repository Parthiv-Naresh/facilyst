import pandas as pd

from facilyst.models import RandomForestRegressor


def test_random_forest_regressor(numeric_features_regression):
    x = pd.DataFrame({"Col_1": [i for i in range(100)]})
    y = pd.Series([i for i in range(100)])

    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x, y)
    rf_predictions = rf_regressor.predict(x)

    assert isinstance(rf_predictions, pd.Series)
    assert len(rf_predictions) == 20

    score = rf_regressor.score(x[80:], y[80:])
    assert isinstance(score, float)

    assert rf_regressor.get_params() == {
        "bootstrap": True,
        "ccp_alpha": 0.0,
        "criterion": "squared_error",
        "max_depth": None,
        "max_features": "auto",
        "max_leaf_nodes": None,
        "max_samples": None,
        "min_impurity_decrease": 0.0,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0,
        "n_estimators": 100,
        "n_jobs": -1,
        "oob_score": False,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
    }
