from facilyst.models import ModelBase


def test_equals():
    class MockPerceptronRegressor(ModelBase):
        name = "Mock Neural Regressor"

        primary_type = "regressor"
        secondary_type = "neural"
        tertiary_type = "perceptron"

        hyperparameters = None

        def get_params(self):
            return {1: "1"}

    class MockTreeRegressor(ModelBase):
        name = "Mock Tree Regressor"
        primary_type = "regressor"
        secondary_type = "ensemble"
        tertiary_type = "tree"

        hyperparameters = None

        def get_params(self):
            return {1: "1"}

    assert MockPerceptronRegressor().__eq__(MockTreeRegressor())
