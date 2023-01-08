from facilyst.models import AutoETSRegressor


def test_auto_ets_time_series_regressor_class_variables():
    assert AutoETSRegressor.name == "AutoETS Regressor"
    assert AutoETSRegressor.primary_type == "time series"
    assert AutoETSRegressor.secondary_type == "regression"
    assert AutoETSRegressor.tertiary_type == "automatic"
    assert list(AutoETSRegressor.hyperparameters.keys()) == [
        "season_length",
        "damped",
    ]
