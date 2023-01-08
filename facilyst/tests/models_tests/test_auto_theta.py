from facilyst.models import AutoThetaRegressor


def test_auto_theta_time_series_regressor_class_variables():
    assert AutoThetaRegressor.name == "AutoTheta Regressor"
    assert AutoThetaRegressor.primary_type == "time series"
    assert AutoThetaRegressor.secondary_type == "regression"
    assert AutoThetaRegressor.tertiary_type == "automatic"
    assert list(AutoThetaRegressor.hyperparameters.keys()) == [
        "season_length",
        "decomposition_type",
        "model",
    ]
