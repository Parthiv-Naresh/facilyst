import os
import pathlib
import re

import pandas as pd
import pytest

from facilyst.graphs import GraphBase, Line, Scatter
from facilyst.mocks import Dates, Features, MockBase, Wave
from facilyst.utils import _get_subclasses
from facilyst.utils.gen_utils import (
    handle_problem_type,
    import_errors_dict,
    import_or_raise,
    infer_problem_type,
    prepare_data,
)

expected_mock_subclasses = [
    Wave,
    Features,
    Dates,
]

expected_graph_subclasses = [
    Scatter,
    Line,
]


def test_mock_get_subclasses():
    actual_mock_subclasses = _get_subclasses(MockBase)
    assert actual_mock_subclasses == expected_mock_subclasses


def test_graph_get_subclasses():
    actual_graph_subclasses = _get_subclasses(GraphBase)
    assert actual_graph_subclasses == expected_graph_subclasses


@pytest.fixture
def current_dir():
    return os.path.dirname(os.path.abspath(__file__))


def test_import_or_raise(has_no_extra_dependencies, current_dir):
    extra_reqs = open(
        os.path.join(
            current_dir, pathlib.Path("..", "..", "..", "extra-requirements.txt")
        )
    ).readlines()[1:]
    extra_reqs = {re.match(r"([a-zA-Z_\-]+)", extra)[0] for extra in extra_reqs}

    with pytest.raises(ImportError, match=f"Missing extra dependency 'facilyst_'"):
        import_or_raise(
            "facilyst_",
        )

    if has_no_extra_dependencies:
        for extra in extra_reqs:
            with pytest.raises(
                ImportError, match=f"Missing extra dependency '{extra}'"
            ):
                import_or_raise(extra, import_errors_dict[extra])
    else:
        _ = [import_or_raise(extra) for extra in extra_reqs]


@pytest.mark.parametrize(
    "problem_type_actual, problem_type_expected, valid",
    [
        ("Regression", "regression", True),
        ("Classification", "classification", True),
        ("classifier", "classification", True),
        ("BINARY", "binary", True),
        ("MultiClass", "multiclass", True),
        ("multi Class", "multiclass", True),
        ("ts regression", "time series", True),
        ("tsr", None, False),
        ("timeseriesregression", None, False),
        ("time series", "time series", True),
    ],
)
def test_handle_problem_type(problem_type_actual, problem_type_expected, valid):

    if valid:
        assert handle_problem_type(problem_type_actual) == problem_type_expected
    else:
        with pytest.raises(
            ValueError,
            match="That problem type isn't recognized!",
        ):
            _ = handle_problem_type(problem_type_actual)


@pytest.mark.parametrize(
    "problem_type_expected", ["regression", "time series", "classification"]
)
def test_infer_problem_type(
    problem_type_expected,
    numeric_features_regression,
    time_series_data,
    numeric_features_multi_classification,
):
    if problem_type_expected == "regression":
        x, y = numeric_features_regression
    elif problem_type_expected == "time series":
        x, _, y, _ = time_series_data()
    elif problem_type_expected == "classification":
        x, y = numeric_features_multi_classification

    assert infer_problem_type(y, x) == problem_type_expected
    if problem_type_expected == "time series":
        problem_type_expected = "regression"
    assert infer_problem_type(y) == problem_type_expected


def test_prepare_data(numeric_features_regression):
    x, y = None, None
    assert prepare_data(x, y) == (None, None)
    x, y = numeric_features_regression
    x_new, y_new = prepare_data(x, y)
    pd.testing.assert_frame_equal(x_new, pd.DataFrame(x))
    pd.testing.assert_series_equal(y_new, pd.Series(y))
    assert x_new.ww.schema is None
    assert y_new.ww.schema is None
    x_new, y_new = prepare_data(x, y, True)
    assert x_new.ww.schema is not None
    assert y_new.ww.schema is not None
