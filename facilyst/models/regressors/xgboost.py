"""A model that uses gradient boosting on decision trees for regression problems."""
from hyperopt import hp
from xgboost import XGBRegressor as xg_regressor

from facilyst.models.model_base import ModelBase


class XGBoostRegressor(ModelBase):
    """The XGBoost Regressor (via xgboost's library).

     This is a regressor that uses gradient boosting on decision trees.

    :param n_estimators: The maximum number of trees that can be built. Defaults to 50.
    :type n_estimators: int, optional
    :param max_depth: The maximum depth of the tree. Defaults to None.
    :type max_depth: int, optional
    :param learning_rate: The learning rate.
    :type learning_rate: float, optional
    """

    name: str = "XGBoost Regressor"

    primary_type: str = "regression"
    secondary_type: str = "ensemble"
    tertiary_type: str = "tree"

    hyperparameters: dict = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "max_depth": hp.randint("max_depth", 2, 10),
        "learning_rate": hp.uniform("learning_rate", 0.001, 1.0),
    }

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = None,
        learning_rate: float = None,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        xg_boost_model = xg_regressor(**parameters)

        super().__init__(model=xg_boost_model, parameters=parameters)
