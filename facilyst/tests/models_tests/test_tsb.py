from facilyst.models import TSBRegressor


def test_tsb_time_series_regressor_class_variables():
    assert TSBRegressor.name == "TSB Regressor"
    assert TSBRegressor.primary_type == "time series"
    assert TSBRegressor.secondary_type == "regression"
    assert TSBRegressor.tertiary_type == "sparse"
    assert list(TSBRegressor.hyperparameters.keys()) == ["alpha_d", "alpha_p"]
