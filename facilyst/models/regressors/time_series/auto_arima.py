"""An automatic autoregressive integrated moving average model for time series regression problems."""
from typing import Optional

from hyperopt import hp
from statsforecast.models import AutoARIMA

from facilyst.models import TimeSeriesModelBase


class AutoARIMARegressor(TimeSeriesModelBase):
    """The Auto ARIMA Regressor (via statsforecast's implementation).

    This is a time series regressor that automatically selects the best ARIMA (AutoRegressive Integrated Moving Average)
    model using an information criterion.

    d (int): Order of first-differencing. Optional.
    D (int): Order of seasonal-differencing. Optional.
    max_p (int): Max autorregresives p.
    max_q (int): Max moving averages q.
    max_P (int): Max seasonal autorregresives P.
    max_Q (int): Max seasonal moving averages Q.
    max_d (int): Max non-seasonal differences.
    max_D (int): Max seasonal differences.
    start_p (int): Starting value of p in stepwise procedure.
    start_q (int): Starting value of q in stepwise procedure.
    start_P (int): Starting value of P in stepwise procedure.
    start_Q (int): Starting value of Q in stepwise procedure.
    stationary (bool): If True, restricts search to stationary models.
    seasonal (bool): If False, restricts search to non-seasonal models.
    ic (str): Information criterion to be used in model selection.
    nmodels (int): Number of models considered in stepwise search.
    season_length (int): Number of observations per unit of time.
    """

    name: str = "AutoARIMA Regressor"

    primary_type: str = "time series"
    secondary_type: str = "regression"
    tertiary_type: str = "automatic"

    hyperparameters: dict = {
        "stationary": hp.choice("stationary", [True, False]),
        "seasonal": hp.choice("seasonal", [True, False]),
        "nmodels": hp.randint("nmodels", 50, 200),
    }

    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = "aicc",
        nmodels: int = 94,
        season_length: int = 1,
        **kwargs,
    ) -> None:
        parameters = {
            "d": d,
            "D": D,
            "max_p": max_p,
            "max_q": max_q,
            "max_P": max_P,
            "max_Q": max_Q,
            "max_d": max_d,
            "max_D": max_D,
            "start_p": start_p,
            "start_q": start_q,
            "start_P": start_P,
            "start_Q": start_Q,
            "stationary": stationary,
            "seasonal": seasonal,
            "ic": ic,
            "nmodels": nmodels,
            "season_length": season_length,
        }
        parameters.update(kwargs)

        autoarima_regressor_model = AutoARIMA(**parameters)

        super().__init__(model=autoarima_regressor_model, parameters=parameters)
