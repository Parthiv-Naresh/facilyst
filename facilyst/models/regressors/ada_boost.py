"""An ensemble weighted model for regression problems."""
from typing import Any, Optional

import woodwork as ww
from hyperopt import hp
from sklearn.ensemble import AdaBoostRegressor as ada_regressor
from sklearn.tree import DecisionTreeRegressor

from facilyst.models.model_base import ModelBase
from facilyst.utils import prepare_data


class ADABoostRegressor(ModelBase):
    """The ADA Boost Regressor (via sklearn's implementation).

    This is an ensemble regressor that fits differently weighted copies of the same regressor on the dataset repeatedly
    depending on the error of subsequent predictions.

    :param base_estimator: The base estimator from which the boosted ensemble is built. Defaults to DecisionTreeRegressor.
    :type base_estimator: object, optional
    :param n_estimators: The maximum number of estimators at which boosting is terminated. Defaults to 50.
    :type n_estimators: int, optional
    :param learning_rate: Weight applied to each regressor at each boosting iteration. A higher learning rate increases
    the contribution of each regressor. There is a trade-off between the learning_rate and n_estimators parameters.
    Defaults to 1.0.
    :type learning_rate: float, optional
    :param loss: The loss function to use when updating the weights after each boosting iteration. Options are `linear`,
    `square`, and `exponential`. Defaults to `linear`.
    :type loss: str, optional
    """

    name: str = "ADA Boost Regressor"

    primary_type: str = "regression"
    secondary_type: str = "ensemble"
    tertiary_type: str = "tree"

    hyperparameters: dict = {
        "n_estimators": hp.choice("n_estimators", [10, 50, 100, 200, 300]),
        "learning_rate": hp.uniform("learning_rate", 0, 1),
        "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    }

    def __init__(
        self,
        base_estimator: Optional[object] = DecisionTreeRegressor(),
        n_estimators: Optional[int] = 50,
        learning_rate: Optional[float] = 1.0,
        loss: Optional[str] = "linear",
        random_state: Optional[int] = 0,
        **kwargs,
    ) -> None:
        parameters = {
            "base_estimator": base_estimator,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "loss": loss,
            "random_state": random_state,
        }
        parameters.update(kwargs)

        ada_regressor_model = ada_regressor(**parameters)

        super().__init__(model=ada_regressor_model, parameters=parameters)

    def fit(self, x_train, y_train) -> Any:
        """Fits ADABoost model to the data.

        :param x_train: The training data for the model to be fitted on.
        :type x_train: pd.DataFrame or np.ndarray
        :param y_train: The training targets for the model to be fitted on.
        :type y_train: pd.Series or np.ndarray
        """
        x_train, y_train = prepare_data(x_train, y_train, True)
        y_train = ww.init_series(y_train)
        x_train = x_train.ww.select(exclude="Categorical")
        super().fit(x_train, y_train)
