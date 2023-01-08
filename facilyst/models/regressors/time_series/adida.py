"""A temporal aggregation model used with sparse data for time series regression problems."""
from statsforecast.models import ADIDA

from facilyst.models import TimeSeriesModelBase


class ADIDARegressor(TimeSeriesModelBase):
    """The Aggregate-Disaggregate Intermittent Demand Approach (ADIDA) Regressor (via statsforecast's implementation).

    This is a time series regressor that uses temporal aggregation to reduce the number of zero observations. Once the
    data has been aggregated, it uses the optimized SES to generate the forecasts at the new level. It then breaks down
    the forecast to the original level using equal weights. ADIDA specializes on sparse or intermittent series with very
    few non-zero observations.

    """

    name: str = "ADIDA Regressor"

    primary_type: str = "time series"
    secondary_type: str = "regression"
    tertiary_type: str = "sparse"

    hyperparameters: dict = {}

    def __init__(
        self,
        **kwargs,
    ) -> None:
        parameters = {}
        parameters.update(kwargs)

        adida_regressor_model = ADIDA(**parameters)

        super().__init__(model=adida_regressor_model, parameters=parameters)
