"""A model that uses gradient boosting on decision trees alongside categorical encoding for regression problems."""
from typing import Optional, Any, Union

import numpy as np
import pandas as pd
import woodwork as ww
from hyperopt import hp

from facilyst.models.model_base import ModelBase
from facilyst.utils import import_errors_dict, import_or_raise, prepare_data


class CatBoostRegressor(ModelBase):
    """The CatBoost Regressor (via catboost's library).

     This is a regressor that uses gradient boosting on decision trees alongside categorical encoding.

    :param n_estimators: The maximum number of trees that can be built. Defaults to 50.
    :type n_estimators: int, optional
    :param max_depth: The maximum depth of the tree. Defaults to None.
    :type max_depth: int, optional
    :param learning_rate: The learning rate.
    :type learning_rate: float, optional
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
            "random_state": random_state,
        }
        parameters.update(kwargs)

        cat_regressor = import_or_raise("catboost", import_errors_dict["catboost"])

        catboost_model = cat_regressor.CatBoostRegressor(**parameters)

        super().__init__(model=catboost_model, parameters=parameters)

    def fit(self, x_train, y_train) -> Any:
        x_train, y_train = prepare_data(x_train, y_train, True)
        self.string_features = x_train.select_dtypes(include=['string']).columns.tolist()
        self.columns_with_nan = x_train.columns[x_train.isna().any()].tolist()
        cols_to_drop = set(self.string_features).union(set(self.columns_with_nan))
        x_train = x_train.drop(cols_to_drop, axis=1)

        x_train, y_train = prepare_data(x_train, y_train, True)
        self.categorical_features = x_train.ww.select(include="Categorical").columns.tolist()

        self.model.fit(x_train, y_train, silent=True, cat_features=self.categorical_features)
        return self

    def predict(self, x_test: Union[pd.DataFrame, np.ndarray]) -> Any:
        x_test, _ = prepare_data(x_test, ww_initialize=True)
        cols_to_drop = set(self.string_features).union(set(self.columns_with_nan))
        x_test = x_test.drop(cols_to_drop, axis=1)

        x_test, _ = prepare_data(x_test, ww_initialize=True)

        return prepare_data(y=self.model.predict(x_test), ww_initialize=True)[1]
