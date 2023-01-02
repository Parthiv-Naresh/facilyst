from facilyst.models import IMAPARegressor


def test_imapa_time_series_regressor_class_variables():
    assert IMAPARegressor.name == "IMAPA Regressor"
    assert IMAPARegressor.primary_type == "regression"
    assert IMAPARegressor.secondary_type == "time series"
    assert IMAPARegressor.tertiary_type == "sparse"
    assert list(IMAPARegressor.hyperparameters.keys()) == []
