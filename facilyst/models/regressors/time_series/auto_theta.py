"""An automatic theta-based and MSE-determined model for time series regression problems."""
from typing import Optional

from hyperopt import hp
from statsforecast.models import AutoTheta

from facilyst.models import TimeSeriesModelBase


class AutoThetaRegressor(TimeSeriesModelBase):
    """The Auto Theta Regressor (via statsforecast's implementation).

    This is a time series regressor that automatically selects the best Theta (Standard Theta Model (‘STM’), Optimized
    Theta Model (‘OTM’), Dynamic Standard Theta Model (‘DSTM’), Dynamic Optimized Theta Model (‘DOTM’)) model using mse.

    season_length (int): Number of observations per unit of time. Optional.
    decomposition_type (str): Seasonal decomposition type, ‘multiplicative’ (default) or ‘additive’.
    model (str): A parameter that "dampens" the trend. Optional.
    """

    name: str = "AutoTheta Regressor"

    primary_type: str = "regression"
    secondary_type: str = "time series"
    tertiary_type: str = "automatic"

    hyperparameters: dict = {
        "season_length": hp.choice("season_length", [1, 4, 7, 12, 24, 30, 52, 60, 365]),
        "decomposition_type": hp.choice(
            "decomposition_type", ["multiplicative", "additive"]
        ),
        "model": hp.choice("model", ["STM", "OTM", "DSTM", "DOTM"]),
    }

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: str = "multiplicative",
        model: Optional[str] = "DOTM",
        **kwargs,
    ) -> None:
        parameters = {
            "season_length": season_length,
            "decomposition_type": decomposition_type,
            "model": model,
        }
        parameters.update(kwargs)

        autotheta_regressor_model = AutoTheta(**parameters)

        super().__init__(model=autotheta_regressor_model, parameters=parameters)
