"""A model that uses gradient boosting on decision trees alongside categorical encoding for regression problems."""
from typing import Optional

from hyperopt import hp

from facilyst.models.model_base import ModelBase
from facilyst.utils import import_errors_dict, import_or_raise


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
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "allow_writing_files": allow_writing_files,
            "random_state": random_state,
        }
        parameters.update(kwargs)

        cat_regressor = import_or_raise("catboost", import_errors_dict["catboost"])

        catboost_model = cat_regressor.CatBoostRegressor(**parameters)

        super().__init__(model=catboost_model, parameters=parameters)
