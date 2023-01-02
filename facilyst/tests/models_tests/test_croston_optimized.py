from facilyst.models import CrostonOptimizedRegressor


def test_croston_optimized_time_series_regressor_class_variables():
    assert CrostonOptimizedRegressor.name == "Croston Optimized Regressor"
    assert CrostonOptimizedRegressor.primary_type == "regression"
    assert CrostonOptimizedRegressor.secondary_type == "time series"
    assert CrostonOptimizedRegressor.tertiary_type == "sparse"
    assert list(CrostonOptimizedRegressor.hyperparameters.keys()) == []
