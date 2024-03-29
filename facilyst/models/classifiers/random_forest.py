"""An ensemble bootstrapping tree-based model for classification problems."""
from typing import Optional

from hyperopt import hp
from sklearn.ensemble import RandomForestClassifier as rf_classifier

from facilyst.models.model_base import ModelBase


class RandomForestClassifier(ModelBase):
    """The Random Forest Classifier (via sklearn's implementation).

     This is an ensemble classifier that fits multiple trees on the data.

    :param n_estimators: The number of trees in the forest. Defaults to 100.
    :type n_estimators: int, optional
    :param max_depth: The maximum depth of the tree. Defaults to no maximum depth, nodes are expanded until all leaves
    are pure or until all leaves contain less than 2 samples.
    :type max_depth: int, optional
    :param criterion: The function to measure the quality of a split. Defaults to `gini`.
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

    name: str = "Random Forest Classifier"

    primary_type: str = "classification"
    secondary_type: str = "ensemble"
    tertiary_type: str = "tree"

    hyperparameters: dict = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "max_depth": hp.randint("max_depth", 2, 10),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.choice("max_features", ["auto", "sqrt"]),
        "ccp_alpha": hp.uniform("ccp_alpha", 0.0, 1.0),
        "max_samples": hp.choice("max_samples", [0.6, 0.75, None]),
    }

    def __init__(
        self,
        n_estimators: Optional[int] = 100,
        max_depth: Optional[int] = None,
        criterion: Optional[str] = "gini",
        max_features: Optional[str] = "auto",
        ccp_alpha: Optional[float] = 0.0,
        max_samples: Optional[int] = None,
        n_jobs: Optional[int] = -1,
        random_state: Optional[int] = 0,
        **kwargs,
    ) -> None:
        parameters = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "criterion": criterion,
            "max_features": max_features,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        parameters.update(kwargs)

        random_forest_model = rf_classifier(**parameters)

        super().__init__(model=random_forest_model, parameters=parameters)
