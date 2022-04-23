import pandas as pd
import pytest

from facilyst.models import CatBoostClassifier


@pytest.mark.parametrize("classification_type", ["binary", "multiclass"])
def test_catboost_classifier(
    classification_type,
    numeric_features_binary_classification,
    numeric_features_multi_classification,
):
    x, y = (
        numeric_features_binary_classification
        if classification_type == "binary"
        else numeric_features_multi_classification
    )

    catboost_classifier = CatBoostClassifier()
    catboost_classifier.fit(x, y)
    catboost_predictions = catboost_classifier.predict(x)

    assert isinstance(catboost_predictions, pd.Series)
    assert len(catboost_predictions) == 100

    score = catboost_classifier.score(x, y)
    assert isinstance(score, float)

    assert catboost_classifier.get_params() == {
        "n_estimators": 50,
    }
