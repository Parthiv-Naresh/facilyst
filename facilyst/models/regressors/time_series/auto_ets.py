"""An automatic exponential smoothing model for time series regression problems."""
from typing import Optional

from hyperopt import hp
from statsforecast.models import AutoETS

from facilyst.models import TimeSeriesModelBase


class AutoETSRegressor(TimeSeriesModelBase):
    """The Auto ETS Regressor (via statsforecast's implementation).

    This is a time series regressor that automatically selects the best ETS (Error, Trend, Seasonality) model using an
    information criterion.

    season_length (int): Number of observations per unit of time. Optional.
    model (str): Controls the state space equations. Options are M (multiplicative), A (additive), Z (optimized), and
        N (ommited).
        The E (error) can be: M, A, or Z
        The T (trend) can be: M, A, Z, or N
        The S (seasonality) can be: M, A, Z, or N
        Defaults to "ZZZ", or optimized for all three terms.
    damped (bool): A parameter that "dampens" the trend. Optional.
    """

    name: str = "AutoETS Regressor"

    primary_type: str = "regression"
    secondary_type: str = "time series"
    tertiary_type: str = "automatic"

    hyperparameters: dict = {
        "season_length": hp.choice("season_length", [1, 4, 7, 12, 24, 30, 52, 60, 365]),
        "damped": hp.choice("damped", [True, False]),
    }

    def __init__(
        self,
        season_length: int = 1,
        model: str = "ZZZ",
        damped: Optional[bool] = None,
        **kwargs,
    ) -> None:
        parameters = {
            "season_length": season_length,
            "model": model,
            "damped": damped,
        }
        parameters.update(kwargs)

        autoets_regressor_model = AutoETS(**parameters)

        super().__init__(model=autoets_regressor_model, parameters=parameters)
