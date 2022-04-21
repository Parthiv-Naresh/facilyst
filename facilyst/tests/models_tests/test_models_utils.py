import pytest

from facilyst.models import (
    ADABoostRegressor,
    BaggingRegressor,
    CatBoostRegressor,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    MultiLayerPerceptronRegressor,
    RandomForestRegressor,
    XGBoostRegressor,
)
from facilyst.models.neural_networks.bert_classifier import BERTBinaryClassifier
from facilyst.models.neural_networks.bert_qa import BERTQuestionAnswering
from facilyst.models.utils import get_models

all_regressors = [
    ADABoostRegressor,
    BaggingRegressor,
    BERTQuestionAnswering,
    CatBoostRegressor,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    MultiLayerPerceptronRegressor,
    RandomForestRegressor,
    XGBoostRegressor,
]

all_classifiers = [
    BERTBinaryClassifier,
]

tree_regressors = [
    ADABoostRegressor,
    BaggingRegressor,
    CatBoostRegressor,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    XGBoostRegressor,
]

nlp_models = [
    BERTBinaryClassifier,
    BERTQuestionAnswering,
]


def test_no_model_name_passed():
    with pytest.raises(ValueError, match="No model name passed."):
        get_models(model="")


def test_model_type_doesnt_exist():
    with pytest.raises(ValueError, match="That model type doesn't exist."):
        get_models(model="nonexistent")


def test_no_model_type_of_problem_type():
    with pytest.raises(
        ValueError,
        match="There are no neural models belong to the classification problem type available.",
    ):
        get_models(model="neural", problem_type="classification")


def test_model_name_doesnt_exist():
    with pytest.raises(ValueError, match="That model doesn't exist."):
        get_models(model="Extra Special Regressor")


def test_no_model_name_of_problem_type():
    with pytest.raises(
        ValueError,
        match="The model Random Forest Regressor was found but doesn't match the problem type classification",
    ):
        get_models(model="Random Forest Regressor", problem_type="classification")


@pytest.mark.parametrize(
    "model, problem_type, expected",
    [
        ("Random Forest Regressor", "regression", [RandomForestRegressor]),
        ("tree", "regression", tree_regressors),
        ("regression", None, all_regressors),
        ("all", None, all_regressors + all_classifiers),
        ("nlp", None, nlp_models)
    ],
)
def test_get_models(model, problem_type, expected):
    actual_models = get_models(model, problem_type)

    assert len(actual_models) == len(expected)
    assert set(actual_models) == set(expected)
