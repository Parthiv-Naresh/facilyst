"""An ensemble bagging model for regression problems."""
from hyperopt import hp
from sklearn.ensemble import BaggingRegressor as bagging_regressor
from sklearn.tree import DecisionTreeRegressor

from facilyst.models.model_base import ModelBase


class BaggingRegressor(ModelBase):
    """The Bagging Regressor (via sklearn's implementation).

    This is an ensemble regressor that fits base regressors on random subsets of the dataset (wuth replacement) and then
    aggregates predictions.

    :param base_estimator: The base estimator from which the boosted ensemble is built. Defaults to DecisionTreeRegressor.
    :type base_estimator: object, optional
    :param n_estimators: The maximum number of estimators at which boosting is terminated. Defaults to 50.
    :type n_estimators: int, optional
    :param max_samples: The number of samples to draw from x to train each base estimator. If an int has been passed,
    it will draw `max_samples` samples. If a float has been passed, it will draw that percentage of samples. Defaults to 1.0.
    :type max_samples: float, optional
    :param oob_score: Whether to use out-of-bag samples to estimate the generalization error. Defaults to False.
    :type oob_score: bool, optional
    """

    name = "ADA Boost Regressor"

    primary_type = "regression"
    secondary_type = "ensemble"
    tertiary_type = "tree"

    hyperparameters = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "max_samples": hp.uniform("max_samples", 0.5, 1.0),
        "oob_score": hp.choice("oob_score", [True, False]),
    }

    def __init__(
        self,
        base_estimator=DecisionTreeRegressor(),
        n_estimators=50,
        max_samples=1.0,
        oob_score=False,
        n_jobs=-1,
        **kwargs,
    ):
        parameters = {
            "base_estimator": base_estimator,
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
        }
        parameters.update(kwargs)

        bag_regressor = bagging_regressor(**parameters)

        super().__init__(model=bag_regressor, parameters=parameters)
