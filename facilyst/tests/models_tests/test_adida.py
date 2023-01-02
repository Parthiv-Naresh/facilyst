from facilyst.models import ADIDARegressor


def test_adida_time_series_regressor_class_variables():
    assert ADIDARegressor.name == "ADIDA Regressor"
    assert ADIDARegressor.primary_type == "regression"
    assert ADIDARegressor.secondary_type == "time series"
    assert ADIDARegressor.tertiary_type == "sparse"
    assert list(ADIDARegressor.hyperparameters.keys()) == []
