import pandas as pd
import pytest

from facilyst.models import TimeSeriesModelBase
from facilyst.models.utils import get_models


def test_time_series_models_warning_horizon_and_x_none(
    time_series_data, mock_time_series_model_class
):
    x_train, x_test, y_train, y_test = time_series_data()

    ts_model = mock_time_series_model_class()

    with pytest.raises(ValueError, match=f"Both horizon and x_test cannot be None."):
        ts_model.fit(y_train)
        ts_model.predict()

    with pytest.raises(ValueError, match=f"Both horizon and x_test cannot be None."):
        ts_model.forecast(y_train=y_train)

    with pytest.raises(
        ValueError,
        match=r"The length of x_test \(20\) is less than the horizon \(35\).",
    ):
        ts_model.forecast(y_train=y_train, horizon=35, x_test=x_test)


@pytest.mark.parametrize("ts_model", get_models("time series"))
def test_get_params(ts_model):
    model = ts_model()
    assert model.get_params() == model.model.__dict__


def test_time_series_models_only_horizon_length_of_x_test_returned(time_series_data):
    x_train, x_test, y_train, y_test = time_series_data()
    new_x_test, horizon = TimeSeriesModelBase._check_for_errors(x_test, 5)
    pd.testing.assert_frame_equal(new_x_test, x_test[:5])
    assert horizon == 5


@pytest.mark.parametrize("make_index_datetime_x", [True, False])
@pytest.mark.parametrize("make_index_datetime_y", [True, False])
@pytest.mark.parametrize("numeric_features", [True, False])
@pytest.mark.parametrize("freq", ["6H", "3D", "M"])
@pytest.mark.parametrize("target_wave", [(12, 3, 3), (5, 2, -4)])
@pytest.mark.parametrize("ts_model", get_models("time series"))
def test_time_series_models_predict_and_forecast_are_equal(
    ts_model,
    target_wave,
    freq,
    numeric_features,
    make_index_datetime_y,
    make_index_datetime_x,
    time_series_data,
):
    x_train, x_test, y_train, y_test = time_series_data(
        make_index_datetime_x=make_index_datetime_x,
        make_index_datetime_y=make_index_datetime_y,
        numeric_features=numeric_features,
        freq=freq,
        target_wave=target_wave,
        num_rows=100,
    )

    ts_regressor = ts_model()

    ts_regressor.fit(y_train=y_train, x_train=x_train)
    ts_predictions = ts_regressor.predict(horizon=len(x_test), x_test=x_test)
    ts_forecasts = ts_regressor.forecast(
        y_train=y_train,
        horizon=len(x_test),
        x_train=x_train,
        x_test=x_test,
    )

    pd.testing.assert_series_equal(ts_predictions, ts_forecasts)
    assert isinstance(ts_predictions, pd.Series)
    assert len(ts_predictions) == 20
    if make_index_datetime_x or make_index_datetime_y:
        assert isinstance(ts_predictions.index, pd.DatetimeIndex)
