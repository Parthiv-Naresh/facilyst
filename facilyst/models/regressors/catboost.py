"""A model that uses gradient boosting on decision trees alongside categorical encoding for regression problems."""
from catboost import CatBoostRegressor as cat_regressor
from hyperopt import hp

from facilyst.models.model_base import ModelBase


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

    name = "Catboost Regressor"

    primary_type = "regression"
    secondary_type = "None"
    tertiary_type = "tree"

    hyperparameters = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "max_depth": hp.randint("max_depth", 2, 10),
        "learning_rate": hp.uniform("learning_rate", 0.001, 1.0),
    }

    def __init__(
        self,
        n_estimators=50,
        max_depth=None,
        learning_rate=None,
        **kwargs,
    ):
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        parameters.update(kwargs)

        catboost_model = cat_regressor(**parameters)

        super().__init__(model=catboost_model, parameters=parameters)
