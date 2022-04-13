import hyperopt.pyll.stochastic
import pytest
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from facilyst.models.optimizers.hyperopt import HyperoptOptimizer
from facilyst.utils import get_dataset


def test_invalid_classifier_regressor_error():
    with pytest.raises(ValueError, match="Either classifier or regressor must be set."):
        HyperoptOptimizer(classifier=None, regressor=None)

    with pytest.raises(
        ValueError, match="Either classifier or regressor must be set, not both."
    ):
        HyperoptOptimizer(classifier="any", regressor="any")


def test_invalid_model_error():
    with pytest.raises(
        ValueError,
        match="The parameter `iterations_per_model` must be either an int or a dict",
    ):
        HyperoptOptimizer(regressor="any", iterations_per_model=50.0)


def test_hyperopt():
    from pprint import pprint

    print()
    x, y = get_dataset("Sensor_Node_ALE")
    print(x)
    print(y)
    space = {
        "n_estimators": hyperopt.hp.choice(
            "n_estimators", [1, 2, 3, 4, 10, 50, 100, 200, 300]
        ),
        "max_depth": hyperopt.hp.randint("max_depth", 1, 15),
        "criterion": hyperopt.hp.choice(
            "criterion", ["squared_error", "absolute_error"]
        ),
    }

    def hyperparameter_tuning(params):
        print("====================")
        print(params)
        params = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "criterion": params["criterion"],
        }
        print(params)
        clf = RandomForestRegressor(**params)
        clf.fit(x[:80], y[:80])
        score = clf.score(x[80:], y[80:])
        return {"loss": -score, "status": hyperopt.STATUS_OK}

    trials = Trials()

    best = fmin(
        fn=hyperparameter_tuning,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        trials=trials,
    )

    print("Best: {}".format(best))
    pprint(space_eval(space, best))


def test_space():
    space = {
        "a": hp.choice("a", [0, 1]),
        "n_estimators": hp.choice(
            "a",
            [(True, 1 + hp.lognormal("c1", 0, 1)), (False, hp.uniform("c2", -10, 10))],
        ),
    }
    print(hyperopt.pyll.stochastic.sample(space))
    space = {
        "n_estimators": hyperopt.hp.choice(
            "n_estimators", [100, 200, 300, 400, 500, 600]
        ),
        "max_depth": hyperopt.hp.choice("max_depth", [1, 15]),
        "criterion": hyperopt.hp.choice(
            "criterion", ["squared_error", "absolute_error"]
        ),
    }
    print(hyperopt.pyll.stochastic.sample(space))
    """params = {
        'n_estimators': params['n_estimators'],
        'max_depth': params['max_depth'],
        'criterion': params['criterion']
    }"""
    x, y = get_dataset("Sensor_Node_ALE")
    clf = RandomForestRegressor(**hyperopt.pyll.stochastic.sample(space))
    clf.fit(x[:80], y[:80])
    score = clf.score(x[80:], y[80:])
    print(score)


def test_mine():
    print()
    x, y = get_dataset("Sensor_Node_ALE")
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

    opt = HyperoptOptimizer(regressor="tree")
    best_model, best_score = opt.optimize(x, y)
    print(best_score)
    print(best_model.get_params())
    best_model.fit(x_train, y_train)
    score = best_model.score(x_test, y_test)
    print(score)
