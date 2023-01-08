from facilyst.models import AutoARIMARegressor


def test_auto_arima_time_series_regressor_class_variables():
    assert AutoARIMARegressor.name == "AutoARIMA Regressor"
    assert AutoARIMARegressor.primary_type == "time series"
    assert AutoARIMARegressor.secondary_type == "regression"
    assert AutoARIMARegressor.tertiary_type == "automatic"
    assert list(AutoARIMARegressor.hyperparameters.keys()) == [
        "stationary",
        "seasonal",
        "nmodels",
    ]
