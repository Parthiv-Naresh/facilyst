import numpy as np
import pandas as pd
import pytest

from facilyst.models import XGBoostClassifier


@pytest.mark.parametrize("classification_type", ["binary", "multiclass"])
def test_xgboost_classifier(
    classification_type,
    numeric_features_binary_classification,
    numeric_features_multi_classification,
):
    x, y = (
        numeric_features_binary_classification
        if classification_type == "binary"
        else numeric_features_multi_classification
    )

    xgboost_classifier = XGBoostClassifier()
    xgboost_classifier.fit(x, y)
    xgboost_predictions = xgboost_classifier.predict(x)

    assert isinstance(xgboost_predictions, pd.Series)
    assert len(xgboost_predictions) == 100

    score = xgboost_classifier.score(x, y)
    assert isinstance(score, float)

    assert xgboost_classifier.get_params() == {
        "base_score": 0.5,
        "booster": "gbtree",
        "colsample_bylevel": 1,
        "colsample_bynode": 1,
        "colsample_bytree": 1,
        "enable_categorical": False,
        "gamma": 0,
        "gpu_id": -1,
        "importance_type": None,
        "interaction_constraints": "",
        "learning_rate": 0.300000012,
        "max_delta_step": 0,
        "max_depth": 6,
        "min_child_weight": 1,
        "missing": np.nan,
        "monotone_constraints": "()",
        "n_estimators": 50,
        "n_jobs": -1,
        "num_parallel_tree": 1,
        "objective": "binary:logistic"
        if classification_type == "binary"
        else "multi:softprob",
        "predictor": "auto",
        "random_state": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "scale_pos_weight": 1 if classification_type == "binary" else None,
        "subsample": 1,
        "tree_method": "exact",
        "use_label_encoder": True,
        "validate_parameters": 1,
        "verbosity": None,
    }
