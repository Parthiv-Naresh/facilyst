"""An ensemble tree-based model for regression problems."""
from hyperopt import hp
from sklearn.ensemble import RandomForestRegressor as rf_regressor

from facilyst.models.model_base import ModelBase


class RandomForestRegressor(ModelBase):
    """The Random Forest Regressor (via sklearn's implementation) is an ensemble regressor that fits multiple trees on the data to learn.

    :param n_estimators: The number of trees in the forest. Defaults to 100.
    :type n_estimators: int, optional
    :param max_depth: The maximum depth of the tree. Defaults to no maximum depth, nodes are expanded until all leaves
    are pure or until all leaves contain less than 2 samples.
    :type max_depth: int, optional
    :param criterion: The function to measure the quality of a split. Defaults to the squared error.
    :type criterion: str, optional
    :param max_features: The number of features to consider when looking for the best split. Defaults to `auto`.
    :type max_features: str, optional
    :param ccp_alpha: Complexity parameter used for Minimal Cost-Complexity Pruning. Defaults to 0.0.
    :type ccp_alpha: float, optional
    :param max_samples: The number of samples to draw from x to train each base estimator. Defaults to None so the entire
    dataset is used for each base estimator.
    :type max_samples: float or None, optional
    :param n_jobs: The number of cores to be used, -1 uses all available cores.
    :type n_jobs: int, optional
    """

    name = "Random Forest Regressor"

    primary_type = "regressor"
    secondary_type = "ensemble"
    tertiary_type = "tree"

    hyperparameters = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "max_depth": hp.randint("max_depth", 2, 10),
        "criterion": hp.choice("criterion", ["squared_error", "poisson"]),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "ccp_alpha": hp.uniform("ccp_alpha", 0.0, 1.0),
        "max_samples": hp.choice("max_samples", [0.6, 0.75, None]),
    }

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        criterion="squared_error",
        max_features="auto",
        ccp_alpha=0.0,
        max_samples=None,
        n_jobs=-1,
    ):
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "criterion": criterion,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "n_jobs": n_jobs,
        }

        random_forest_model = rf_regressor(**parameters)

        super().__init__(model=random_forest_model)
