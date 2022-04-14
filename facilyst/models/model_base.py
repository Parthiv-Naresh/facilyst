"""Base class for all models."""
from abc import ABC, abstractmethod

import pandas as pd


class ModelBase(ABC):
    """Base initialization for all models.

    :param model: The model to be used.
    :type model: object
    """

    def __init__(self, model=None, parameters=None):
        self.model = model
        self.parameters = parameters

    def __eq__(self, other):
        if not isinstance(other, ModelBase):
            return NotImplemented

        return self.get_params() == other.get_params()

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

    def fit(self, x_train, y_train):
        """Fits model to the data.

        :param x_train: The training data for the model to be fitted on.
        :type x_train: pd.DataFrame or np.ndarray
        :param y_train: The training targets for the model to be fitted on.
        :type y_train: pd.Series or np.ndarray
        """
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x_test):
        """Predicts on the data using the model.

        :param x_test: The testing data for the model to predict on.
        :type x_test: pd.DataFrame or np.ndarray
        """
        predictions = pd.Series(self.model.predict(x_test))
        return predictions

    def score(self, x_test, y_actual):
        """Scores the predictions of the model using R2.

        :param x_test: The testing data for the model to predict on.
        :type x_test: pd.DataFrame or np.ndarray
        :param y_actual: The actual target values to score against.
        :type y_actual: pd.Series or np.ndarray
        """
        score = self.model.score(x_test, y_actual)
        return score

    def get_params(self):
        """Gets the parameters for the model."""
        model_params = self.model.get_params(deep=True)
        return model_params
