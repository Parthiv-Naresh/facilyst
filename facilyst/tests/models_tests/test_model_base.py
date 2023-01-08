import numpy as np
import pandas as pd
import pytest

from facilyst.models import CatBoostClassifier, ModelBase
from facilyst.models.utils import get_models


def test_models_equivalency(mock_regression_model_class):
    mock_class_1 = mock_regression_model_class()
    mock_class_2 = mock_regression_model_class()

    assert mock_class_1 == mock_class_2

    mock_class_1 = mock_regression_model_class(first_arg=4)
    mock_class_2 = mock_regression_model_class(first_arg=1)

    assert not mock_class_1 == mock_class_2

    mock_class_1 = mock_regression_model_class()
    mock_class_1.name = "mock Regression Model"
    mock_class_2 = mock_regression_model_class()

    assert not mock_class_1 == mock_class_2


@pytest.mark.parametrize(
    "regression_model",
    sorted(get_models("regression", exclude="neural"), key=lambda x: x.name),
)
def test_estimators_regressors_sk_equivalent(
    regression_model, numeric_features_regression
):
    x, y = numeric_features_regression

    try:
        regressor = regression_model()
    except ImportError:
        pytest.skip("Skipping test because extra dependencies not installed")
    regressor.fit(x, y)
    dt_predictions = regressor.predict(x)

    sk_dt_regressor = regression_model().model
    sk_dt_regressor.fit(x, y)
    sk_dt_predictions = sk_dt_regressor.predict(x)

    np.testing.assert_array_almost_equal(dt_predictions.values, sk_dt_predictions)

    assert isinstance(dt_predictions, pd.Series)
    assert len(dt_predictions) == 100

    score = regressor.score(x, y)
    assert isinstance(score, float)


@pytest.mark.parametrize(
    "classifier_model",
    sorted(get_models("classification", exclude="neural"), key=lambda x: x.name),
)
def test_estimators_classifiers_sk_equivalent(
    classifier_model, numeric_features_multi_classification
):
    x, y = numeric_features_multi_classification

    try:
        classifier = classifier_model()
    except ImportError:
        pytest.skip("Skipping test because extra dependencies not installed")
    classifier.fit(x, y)
    dt_predictions = classifier.predict(x)

    sk_dt_classifier = classifier_model().model
    sk_dt_classifier.fit(x, y)
    sk_dt_predictions = sk_dt_classifier.predict(x)

    if isinstance(classifier, CatBoostClassifier):
        sk_dt_predictions = pd.Series(sk_dt_predictions.flatten())

    np.testing.assert_array_almost_equal(dt_predictions.values, sk_dt_predictions)

    assert isinstance(dt_predictions, pd.Series)
    assert len(dt_predictions) == 100

    score = classifier.score(x, y)
    assert isinstance(score, float)
