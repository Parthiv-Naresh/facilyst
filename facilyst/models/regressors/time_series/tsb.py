"""A decomposition, demand probability-based model used with sparse data for time series regression problems."""
from hyperopt import hp
from statsforecast.models import TSB

from facilyst.models import TimeSeriesModelBase


class TSBRegressor(TimeSeriesModelBase):
    """The Teunter-Syntetos-Babai (TSB) Regressor (via statsforecast's implementation).

    This is a time series regressor that is a modification of Croston’s method that replaces the inter-demand intervals
    with the demand probability d(t), which is defined as follows:
        1 - if demand occurs at time t
        0 - otherwise
    The forecast is given by d(t) * z(t). Both d(t) and z(t) are forecasted using SES. The smoothing parameters of each
    may differ, like in the optimized Croston’s method. The TSB model specialize on sparse or intermittent series with
    very few non-zero observations.

    alpha_d (float): Smoothing parameter for demand.
    alpha_p (float): Smoothing parameter for probability.
    """

    name: str = "TSB Regressor"

    primary_type: str = "time series"
    secondary_type: str = "regression"
    tertiary_type: str = "sparse"

    hyperparameters: dict = {
        "alpha_d": hp.uniform("alpha_d", 0.05, 0.4),
        "alpha_p": hp.uniform("alpha_p", 0.05, 0.4),
    }

    def __init__(
        self,
        alpha_d: float = 0.2,
        alpha_p: float = 0.2,
        **kwargs,
    ) -> None:
        parameters = {"alpha_d": alpha_d, "alpha_p": alpha_p}
        parameters.update(kwargs)

        tsb_regressor_model = TSB(**parameters)

        super().__init__(model=tsb_regressor_model, parameters=parameters)
