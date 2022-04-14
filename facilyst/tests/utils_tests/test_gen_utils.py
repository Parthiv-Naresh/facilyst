import pytest

from facilyst.graphs import GraphBase, Line, Scatter
from facilyst.mocks import Dates, Features, MockBase, Wave
from facilyst.utils import _get_subclasses
from facilyst.utils.gen_utils import handle_problem_type

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


@pytest.mark.parametrize(
    "problem_type_actual, problem_type_expected, valid",
    [
        ("Regression", "regression", True),
        ("Classification", "classification", True),
        ("classifier", "classification", True),
        ("BINARY", "binary", True),
        ("MultiClass", "multiclass", True),
        ("multi Class", "multiclass", True),
        ("ts regression", "time series regression", True),
        ("tsr", None, False),
        ("timeseriesregression", None, False),
        ("time series", None, False),
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
