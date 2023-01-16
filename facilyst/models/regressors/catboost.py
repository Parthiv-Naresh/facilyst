"""A model that uses gradient boosting on decision trees alongside categorical encoding for regression problems."""
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from hyperopt import hp

from facilyst.models.model_base import ModelBase
from facilyst.utils import import_errors_dict, import_or_raise, prepare_data


class CatBoostRegressor(ModelBase):
    """The CatBoost Regressor (via catboost's library).

     This is a regressor that uses gradient boosting on decision trees alongside categorical encoding.

    n_estimators (int): The maximum number of trees that can be built. Defaults to 50.
    max_depth (int): The maximum depth of the tree. Defaults to None.
    learning_rate (float): The learning rate.
    allow_writing_files (bool): Whether to allow the generation of files during training. Defaults to False.
    """

    name: str = "Catboost Regressor"

    primary_type: str = "regression"
    secondary_type: str = "None"
    tertiary_type: str = "tree"

    hyperparameters: dict = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "max_depth": hp.randint("max_depth", 2, 10),
        "learning_rate": hp.uniform("learning_rate", 0.001, 1.0),
    }

    def __init__(
        self,
        n_estimators: Optional[int] = 50,
        max_depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        allow_writing_files: Optional[bool] = False,
        random_state: Optional[int] = 0,
        **kwargs,
    ) -> None:
        self.columns_with_nan = None
        self.string_features = None
        self.categorical_features = None
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "allow_writing_files": allow_writing_files,
            "random_state": random_state,
        }
        parameters.update(kwargs)

        cat_regressor = import_or_raise("catboost", import_errors_dict["catboost"])

        catboost_model = cat_regressor.CatBoostRegressor(allow_writing_files=False, **parameters)

        super().__init__(model=catboost_model, parameters=parameters)

    def fit(self, x_train, y_train) -> Any:
        """Fits CatBoost Regressor model to the data.

        :param x_train: The training data for the model to be fitted on.
        :type x_train: pd.DataFrame or np.ndarray
        :param y_train: The training targets for the model to be fitted on.
        :type y_train: pd.Series or np.ndarray
        """
        x_train, y_train = prepare_data(x_train, y_train, True)
        self.string_features = x_train.select_dtypes(
            include=["string"]
        ).columns.tolist()
        self.columns_with_nan = x_train.columns[x_train.isna().any()].tolist()
        cols_to_drop = set(self.string_features).union(set(self.columns_with_nan))
        x_train = x_train.drop(cols_to_drop, axis=1)

        x_train, y_train = prepare_data(x_train, y_train, True)
        self.categorical_features = x_train.ww.select(
            include="Categorical"
        ).columns.tolist()

        self.model.fit(
            x_train, y_train, silent=True, cat_features=self.categorical_features
        )
        return self

    def predict(self, x_test: Union[pd.DataFrame, np.ndarray]) -> pd.Series:
        """Predicts on the data using the CatBoost Regressor.

        x_test (pd.DataFrame or np.ndarray): The testing data for the model to predict on.
        return (pd.Series): The predictions.
        """
        x_test, _ = prepare_data(x_test, ww_initialize=True)
        cols_to_drop = set(self.string_features).union(set(self.columns_with_nan))
        x_test = x_test.drop(cols_to_drop, axis=1)

        x_test, _ = prepare_data(x_test, ww_initialize=True)

        return prepare_data(y=self.model.predict(x_test), ww_initialize=True)[1]
