"""Base class for all time series models."""
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from facilyst.models import ModelBase


class TimeSeriesModelBase(ModelBase):
    """Base initialization for all time series models.

    model (object): The model to be used.
    """

    def __init__(
        self, model: Optional[Any] = None, parameters: Optional[dict] = None
    ) -> None:
        self.model = model
        self.parameters = parameters
        self.frequency = None
        self.final_training_index = 0
        super().__init__(model=self.model, parameters=self.parameters)

    def __eq__(self, other) -> bool:
        if not isinstance(other, TimeSeriesModelBase):
            return NotImplemented

        return self.get_params() == other.get_params()

    def _store_final_training_index(
        self, y_train, x_train
    ) -> Union[pd.DataFrame, np.ndarray, None]:
        # pytype: disable=attribute-error
        if isinstance(x_train, pd.DataFrame) and isinstance(
            x_train.index, pd.DatetimeIndex
        ):
            self.final_training_index = x_train.index[-1]
            self.frequency = pd.infer_freq(x_train.index)
        elif isinstance(y_train, pd.Series) and isinstance(
            y_train.index, pd.DatetimeIndex
        ):
            self.final_training_index = y_train.index[-1]
            self.frequency = pd.infer_freq(y_train.index)
        else:
            self.final_training_index = len(y_train) - 1
            self.frequency = None
        if x_train is not None and x_train.size == 0:
            x_train = None
        # pytype: enable=attribute-error
        return x_train

    @staticmethod
    def _convert_data(y, x):
        # pytype: disable=attribute-error
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        # pytype: enable=attribute-error
        return y, x

    def fit(
        self,
        y_train: Union[pd.Series, np.ndarray],
        x_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> Any:
        """Fits time series model to the data.

        y_train (pd.Series or np.ndarray): The training targets for the time series model to be fitted on.
        x_train (pd.DataFrame or np.ndarray): The training data for the time series model to be fitted on. Optional.
        """
        x_train = self._store_final_training_index(y_train, x_train)
        y_train, x_train = TimeSeriesModelBase._convert_data(y=y_train, x=x_train)
        self.model.fit(y=y_train, X=x_train)
        return self

    def _create_predict_index(self, horizon=None):
        if self.frequency is None:
            predict_index = pd.RangeIndex(
                self.final_training_index + 1, self.final_training_index + horizon + 1
            )
        else:
            predict_index = pd.date_range(
                self.final_training_index, freq=self.frequency, periods=horizon + 1
            )
            predict_index = predict_index[1:]
        return predict_index

    def _get_forecast_index(self, x_test, horizon):
        if isinstance(x_test, pd.DataFrame):
            predictions_index = x_test.index  # noqa
            x_test = x_test.to_numpy()  # noqa
        else:
            predictions_index = self._create_predict_index(horizon)
        return predictions_index

    @staticmethod
    def _check_for_errors(x_test, horizon):
        if horizon is None:
            if x_test is None:
                raise ValueError("Both horizon and x_test cannot be None.")
            horizon = len(x_test)
        else:
            if x_test is not None:
                if len(x_test) < horizon:
                    raise ValueError(
                        f"The length of x_test ({len(x_test)}) is less than the horizon ({horizon})."
                    )
                else:
                    x_test = x_test[:horizon]
        return x_test, horizon

    def predict(
        self,
        horizon: Optional[int] = None,
        x_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> pd.Series:
        """Predicts on the data using the time series model.

        horizon (int): The forecast horizon. Will be inferred from the length of the x_test if none is passed.
        x_test (pd.DataFrame or np.ndarray): The testing data for the time series model to predict on. Must be provided
            if horizon is not passed.
        return (pd.Series): The predictions.
        """
        _, x_test = TimeSeriesModelBase._convert_data(y=None, x=x_test)
        x_test, horizon = TimeSeriesModelBase._check_for_errors(x_test, horizon)
        predictions_index = self._get_forecast_index(x_test=x_test, horizon=horizon)
        forecasts_dict = self.model.predict(h=horizon, X=x_test)
        predictions = pd.Series(forecasts_dict["mean"], index=predictions_index)

        return predictions

    def forecast(
        self,
        y_train: Union[pd.Series, np.ndarray],
        horizon: Optional[int] = None,
        x_train: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        x_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> pd.Series:
        """Analogous to fit_predict without storing data in memory.

        y_train (pd.Series or np.ndarray): The training targets for the time series model to be fitted on.
        horizon (int): The forecast horizon. Will be inferred from the length of the x_test if none is passed.
        x_train (pd.DataFrame or np.ndarray): The training data for the time series model to be fitted on. Optional.
        x_test (pd.DataFrame or np.ndarray): The testing data for the time series model to predict on. Must be provided
            if horizon is not passed.
        return (pd.Series): The predictions.
        """
        self._store_final_training_index(y_train, x_train)
        y_train, x_train = TimeSeriesModelBase._convert_data(y=y_train, x=x_train)
        _, x_test = TimeSeriesModelBase._convert_data(y=None, x=x_test)
        x_test, horizon = TimeSeriesModelBase._check_for_errors(x_test, horizon)

        predictions_index = self._get_forecast_index(x_test=x_test, horizon=horizon)
        forecasts_dict = self.model.forecast(
            y=y_train, h=horizon, X=x_train, X_future=x_test
        )
        predictions = pd.Series(forecasts_dict["mean"], index=predictions_index)
        return predictions

    def get_params(self) -> dict:
        """Gets the parameters for the time series model.

        :return: The time series model's parameters.
        :rtype dict:
        """
        model_params = self.model.__dict__
        return model_params
