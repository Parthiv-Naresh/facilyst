from facilyst.models import ModelBase


def test_equals(dummy_perceptron_regressor, dummy_tree_regressor):

    assert dummy_perceptron_regressor().__eq__(dummy_tree_regressor())
