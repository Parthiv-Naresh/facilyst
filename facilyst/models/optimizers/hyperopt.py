"""An optimizer used for hyperparameter tuning via Bayesian optimization."""
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import train_test_split

from facilyst.models import ModelBase
from facilyst.models.utils import get_models
from facilyst.utils import infer_problem_type


class HyperoptOptimizer:
    """Hyperopt is a distributed, asynchronous hyperparameter optimizer .

    For either the `classifier` or `regressor` parameters, the name of the model can be passed or a type of model
    can be specified, which will use all models belonging to that type. For example, `Random Forest Classifier` will
    result in only that model being used, while `tree` will result in all tree-based models being used during optimization.
    If `any` is passed, then every model belonging to that problem type will be used during optimization.
    Either the classifier OR the `regressor` parameter must be set.

    models (str, list, set, ModelBase): The model(s) to use. If a value is passed it must be either a string of a class
    name, describing a model type, or a non-instance model class. If a list or set is passed, each value must be a string
    or a non-instance model class. If None is passed, then the problem type will be inferred and all models belonging to
    that type will be used.
    split (float): The percentage of the data passed to be kept aside for training. 1 - the value set for this parameter
    will be used for testing.
    iterations_per_model (int, dict): The number of iterations that should be performed per model. If a dict is passed, keys should represent
    the name of the model and values should be the number of iterations. If more models are selected than those specified
    in the dict, then they will be set to a default number of iterations of 50.
    """

    name: str = "Hyperopt Optimizer"

    def __init__(
        self,
        models: Optional[Any] = None,
        split: Optional[float] = 0.8,
        iterations_per_model: Optional[Union[int, dict]] = 50,
    ) -> None:
        self.models = HyperoptOptimizer._collect_models(models=models, y=None)
        self.split = split
        self.iterations_per_model = self._collect_iterations(iterations_per_model)
        self.results = {}

        if not isinstance(self.iterations_per_model, (int, dict)):
            raise ValueError(
                "The parameter `iterations_per_model` must be either an int or a dict specifying the "
                "number of iterations per model."
            )

        self.space = self.hyperparameter_space()

    @staticmethod
    def _collect_models(
        models: Optional[Any] = None, y: Optional[pd.Series] = None
    ) -> Optional[set]:
        """Collect all models requested for the optimizer.

        y (pd.Series): Target data.
        return (set): Set of collected models.
        """
        if models is None:
            if y is not None:
                problem_type = infer_problem_type(y)
                return get_models(problem_type=problem_type)
            else:
                return None
        else:
            if isinstance(models, str):
                return get_models(name_or_tag=models)
            elif isinstance(models, (list, set)):
                collected_models = []
                for model in models:
                    if isinstance(model, str):
                        try:
                            tagged_models = get_models(name_or_tag=model)
                            collected_models.extend(tagged_models)
                        except ValueError:
                            raise ValueError(
                                f"The model name `{model}` doesn't correspond to any model name or tag."
                            )
                    elif issubclass(model, ModelBase):
                        collected_models.append(model)
                    else:
                        raise ValueError(
                            f"The model `{model}` is not an accepted type."
                        )
                return set(collected_models)
            elif issubclass(models, ModelBase):
                return {models}
            else:
                raise ValueError(f"The model `{models}` is not an accepted type.")

    def _collect_iterations(self, num_iterations: Union[int, dict]):
        iterations_per_model = {}
        if self.models is None:
            return iterations_per_model
        if isinstance(num_iterations, int):
            for model in self.models:
                iterations_per_model[model.name] = num_iterations
        else:
            for model_name, iterations in num_iterations.items():
                try:
                    if isinstance(model_name, str):
                        matched_model = list(get_models(name_or_tag=model_name))[0]
                    else:
                        matched_model = model_name
                except ValueError:
                    raise ValueError(
                        f"The model name `{model_name}` doesn't correspond to any model name."
                    )
                iterations_per_model[matched_model.name] = iterations
            for model in self.models:
                if model.name not in iterations_per_model:
                    iterations_per_model[model.name] = 50
        return iterations_per_model

    def hyperparameter_space(self) -> list:
        """The collected hyperparameter space for all models selected to search through.

        :rtype list:
        """
        space = []
        if self.models is None:
            return space
        for each_model in self.models:
            model_space = {each_model.name: each_model.hyperparameters}
            space.append(model_space)
        return space

    @staticmethod
    def organize_results(results_dict: dict) -> pd.DataFrame:
        return pd.DataFrame.from_dict(results_dict, orient="index")

    def optimize(
        self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> Tuple[ModelBase, float]:
        """Convenience function to start optimization job and iterate over collected models.

        :param x: All feature data.
        :type x: pd.DataFrame or np.ndarray
        :param y: All target data.
        :type y: pd.Series or np.ndarray
        :return: The best model selected with the corresponding best hyperparameters, and the score achieved.
        :rtype tuple: object, float
        """
        if self.models is None:
            self.models = HyperoptOptimizer._collect_models(models=None, y=y)
            self.space = self.hyperparameter_space()

        results_dict = {}
        for model in self.models:
            results_dict[model.name] = self._optimize(
                x, y, model, model.hyperparameters
            )
        self.results = HyperoptOptimizer.organize_results(results_dict)

        best_model_row = self.results.best_score.idxmin()
        best_model = get_models(name_or_tag=best_model_row)
        best_model = list(best_model)[0]
        best_model_hyperparameters = self.results.loc[best_model_row].best_hyperparameters
        best_score = self.results.loc[best_model_row].best_score

        return best_model(**best_model_hyperparameters), best_score

    def _optimize(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        model: ModelBase,
        space: dict,
    ) -> dict:
        """Optimization per model over hyperparameter space."""

        def cost_function(parameters: dict) -> dict:
            """Cost function definition."""
            parameters = {hyp: parameters[hyp] for hyp in parameters.keys()}

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=self.split
            )
            model_ = model(**parameters)  # pytype: disable=not-callable
            model_.fit(x_train, y_train)
            score = model_.score(x_test, y_test)
            return {"loss": -score, "status": STATUS_OK}

        trials = Trials()

        model_iter = None
        if isinstance(self.iterations_per_model, dict):
            model_iter = self.iterations_per_model.get(model.name, 50)

        best_hyp = fmin(
            fn=cost_function,
            space=space,
            algo=tpe.suggest,
            max_evals=model_iter or 50,
            trials=trials,
            verbose=False,
            allow_trials_fmin=False
        )

        best_trial = trials.best_trial
        best_dict = {
            "best_hyperparameters": space_eval(model.hyperparameters, best_hyp),
            "best_score": round(best_trial["result"]["loss"], 3),
        }

        return best_dict
