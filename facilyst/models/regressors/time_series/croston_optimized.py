"""A decomposition model used with sparse data for time series regression problems."""
from statsforecast.models import CrostonOptimized

from facilyst.models import TimeSeriesModelBase


class CrostonOptimizedRegressor(TimeSeriesModelBase):
    """The Croston Optimized Regressor (via statsforecast's implementation).

    This is a time series regressor that decomposes the original time series into a non-zero demand size z(t) and
    inter-demand intervals p(t). Then the forecast is given by z(t)/p(t). Croston-based models specialize on sparse or
    intermittent series with very few non-zero observations.

    """

    name: str = "Croston Optimized Regressor"

    primary_type: str = "regression"
    secondary_type: str = "time series"
    tertiary_type: str = "sparse"

    hyperparameters: dict = {}

    def __init__(
        self,
        **kwargs,
    ) -> None:
        parameters = {}
        parameters.update(kwargs)

        croston_opt_regressor_model = CrostonOptimized(**parameters)

        super().__init__(model=croston_opt_regressor_model, parameters=parameters)
