"""A multiple temporal aggregation model used with sparse data for time series regression problems."""
from statsforecast.models import IMAPA

from facilyst.models import TimeSeriesModelBase


class IMAPARegressor(TimeSeriesModelBase):
    """The Intermittent Multiple Aggregation Prediction Algorithm (IMAPA) Regressor (via statsforecast's implementation).

    This is a time series regressor that is similar to ADIDA, but instead of using a single aggregation level, it
    considers multiple in order to capture different dynamics of the data. Uses the optimized SES to generate the
    forecasts at the new levels and then combines them using a simple average. The IMAPA model specialize on sparse or
    intermittent series with very few non-zero observations.

    """

    name: str = "IMAPA Regressor"

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

        imapa_regressor_model = IMAPA(**parameters)

        super().__init__(model=imapa_regressor_model, parameters=parameters)
