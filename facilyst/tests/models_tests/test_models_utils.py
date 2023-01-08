from typing import Any, List

import pytest

from facilyst.models import (
    ADABoostClassifier,
    ADABoostRegressor,
    ADIDARegressor,
    AutoARIMARegressor,
    AutoETSRegressor,
    AutoThetaRegressor,
    BaggingClassifier,
    BaggingRegressor,
    CatBoostClassifier,
    CatBoostRegressor,
    CrostonOptimizedRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    IMAPARegressor,
    MultiLayerPerceptronClassifier,
    MultiLayerPerceptronRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    TSBRegressor,
    XGBoostClassifier,
    XGBoostRegressor,
)
from facilyst.models.neural_networks.bert_classifier import (
    BERTBinaryClassifier,
)
from facilyst.models.neural_networks.bert_qa import BERTQuestionAnswering
from facilyst.models.utils import get_models

all_time_series_regressors: List[Any] = [
    ADIDARegressor,
    AutoARIMARegressor,
    AutoETSRegressor,
    AutoThetaRegressor,
    CrostonOptimizedRegressor,
    IMAPARegressor,
    TSBRegressor,
]

non_time_series_regressors: List[Any] = [
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

all_regressors: List[Any] = non_time_series_regressors + all_time_series_regressors

all_classifiers: List[Any] = [
    ADABoostClassifier,
    BaggingClassifier,
    BERTBinaryClassifier,
    CatBoostClassifier,
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    MultiLayerPerceptronClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
]

tree_regressors: List[Any] = [
    ADABoostRegressor,
    BaggingRegressor,
    CatBoostRegressor,
    DecisionTreeRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
    XGBoostRegressor,
]

tree_classifiers: List[Any] = [
    ADABoostClassifier,
    BaggingClassifier,
    CatBoostClassifier,
    DecisionTreeClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    XGBoostClassifier,
]

all_tree_models: List[Any] = tree_regressors + tree_classifiers

nlp_models: List[Any] = [
    BERTBinaryClassifier,
    BERTQuestionAnswering,
]

neural_models = [
    BERTBinaryClassifier,
    BERTQuestionAnswering,
    MultiLayerPerceptronClassifier,
    MultiLayerPerceptronRegressor,
]


def test_no_models_found_error():
    with pytest.raises(ValueError, match="No models were found"):
        get_models(name_or_tag="something", problem_type=None)

    with pytest.raises(ValueError, match="No models were found"):
        get_models(name_or_tag="something", problem_type="regression")


@pytest.mark.parametrize(
    "model, problem_type, exclude, expected",
    [
        ("Random Forest Regressor", "regression", None, [RandomForestRegressor]),
        ("Random Forest", "ALL", None, [RandomForestRegressor, RandomForestClassifier]),
        ("Random Forest", "ALL", "regression", [RandomForestClassifier]),
        ("tree", "regression", None, tree_regressors),
        (
            "tree",
            None,
            "random forest",
            set(all_tree_models) - {RandomForestRegressor, RandomForestClassifier},
        ),
        ("classification", None, "", all_classifiers),
        (None, "Classification", "cat", set(all_classifiers) - {CatBoostClassifier}),
        (None, "regression", None, non_time_series_regressors),
        ("regression", None, None, non_time_series_regressors),
        (
            "regression",
            None,
            "neural",
            set(non_time_series_regressors)
            - {BERTQuestionAnswering, MultiLayerPerceptronRegressor},
        ),
        (None, None, None, all_regressors + all_classifiers),
        ("nlp", None, None, nlp_models),
        ("time series", None, None, all_time_series_regressors),
        ("ets", "time series", None, [AutoETSRegressor]),
    ],
)
def test_get_models(model, problem_type, exclude, expected):
    actual_models = get_models(model, problem_type, exclude)

    assert actual_models == set(expected)
