"""A model that uses gradient boosting on decision trees for classification problems."""
from typing import Optional

from hyperopt import hp

from facilyst.models.model_base import ModelBase
from facilyst.utils import import_errors_dict, import_or_raise


class XGBoostClassifier(ModelBase):
    """The XGBoost Classifier (via xgboost's library).

     This is a classifier that uses gradient boosting on decision trees.

    :param n_estimators: The maximum number of trees that can be built. Defaults to 50.
    :type n_estimators: int, optional
    :param max_depth: The maximum depth of the tree. Defaults to None.
    :type max_depth: int, optional
    :param learning_rate: The learning rate.
    :type learning_rate: float, optional
    """

    name: str = "XGBoost Classifier"

    primary_type: str = "classification"
    secondary_type: str = "ensemble"
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
        n_jobs: Optional[int] = -1,
        random_state: Optional[int] = 0,
        **kwargs,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        parameters.update(kwargs)

        xg_classifier = import_or_raise("xgboost", import_errors_dict["xgboost"])

        xg_boost_model = xg_classifier.XGBClassifier(**parameters)

        super().__init__(model=xg_boost_model, parameters=parameters)
