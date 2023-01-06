"""Base class for all models."""
from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar, Union

import numpy as np
import pandas as pd


class ModelBase(ABC):
    """Base initialization for all models.

    model (object): The model to be used.
    """

    def __init__(
        self, model: Optional[Any] = None, parameters: Optional[dict] = None
    ) -> None:
        self.model = model
        self.parameters = parameters

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelBase):
            return False
        name_equal = self.name == other.name
        primary_equal = self.primary_type == other.primary_type
        secondary_equal = self.secondary_type == other.secondary_type
        tertiary_equal = self.tertiary_type == other.tertiary_type
        params_equal = self.parameters == other.parameters
        return (
            name_equal
            and primary_equal
            and secondary_equal
            and tertiary_equal
            and params_equal
        )

    @property
    @abstractmethod
    def name(self):
        """Name of the model."""

    @property
    @abstractmethod
    def primary_type(self):
        """Primary type of the model."""

    @property
    @abstractmethod
    def secondary_type(self):
        """Secondary type of the model."""

    @property
    @abstractmethod
    def tertiary_type(self):
        """Tertiary type of the model."""

    @property
    @abstractmethod
    def hyperparameters(self):
        """Hyperparameter space for the model."""

    def fit(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> Any:
        """Fits model to the data.

        :param x_train: The training data for the model to be fitted on.
        :type x_train: pd.DataFrame or np.ndarray
        :param y_train: The training targets for the model to be fitted on.
        :type y_train: pd.Series or np.ndarray
        """
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Predicts on the data using the model.

        x_test (pd.DataFrame or np.ndarray): The testing data for the model to predict on.
        return (pd.Series): The predictions.
        """
        predictions = pd.Series(self.model.predict(x_test))
        return predictions

    def score(
        self,
        x_test: Union[pd.DataFrame, np.ndarray],
        y_actual: Union[pd.Series, np.ndarray],
    ) -> float:
        """Scores the predictions of the model using R2.

        :param x_test: The testing data for the model to predict on.
        :type x_test: pd.DataFrame or np.ndarray
        :param y_actual: The actual target values to score against.
        :type y_actual: pd.Series or np.ndarray
        :return: Calculated score.
        :rtype float:
        """
        score = self.model.score(x_test, y_actual)
        return score

    def get_params(self) -> dict:
        """Gets the parameters for the model.

        :return: The model's parameters.
        :rtype dict:
        """
        model_params = self.model.get_params(deep=True)
        return model_params
