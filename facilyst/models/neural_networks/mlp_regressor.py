"""A multi-perceptron neural network model."""
from sklearn.neural_network import MLPRegressor

from facilyst.models.model_base import ModelBase


class MultiLayerPerceptronRegressor(ModelBase):
    """The Multilayer Perceptron regressor is a feedforward neural network made of hidden layers.

    :param hidden_layer_sizes: The number of neurons in each hidden layer. For example, (34, 78, 90) results in 3 middle
    layers with 34, 78, and 90 neurons respectively.
    :type hidden_layer_sizes: tuple, optional
    :param activation: Activation function for the hidden layers. Options are 'identity', 'logistic', 'tanh', and 'relu'.
    :type activation: str, optional
    :param solver: The solver for weight optimization. Options are `lbfgs`, `sgd`, and `adam`.
    :type solver: str, optional
    :alpha alpha: L2 penalty (regularization term) parameter.
    :type alpha: float, optional
    :param batch_size: Size of minibatches for stochastic optimizers. Auto sets the batch_size to min(200, n_samples).
    :type batch_size: int, optional
    :param learning_rate: Learning rate schedule for weight updates. Options are `constant`, `invscaling`, and `adaptive`.
    :type learning_rate: str, optional
    :param learning_rate_init: The initial learning rate used. It controls the step-size in updating the weights. Only
    used when solver=’sgd’ or ‘adam’.
    :type learning_rate_init: float, optional
    :param max_iter: Maximum number of iterations.
    :type max_iter: int, optional
    """

    name = "Multilayer Perceptron"

    hyperparameters = {}

    primary_type = "regression"
    secondary_type = "neural"
    tertiary_type = "perceptron"

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=200,
        **kwargs,
    ):

        parameters = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "solver": solver,
            "alpha": alpha,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "learning_rate_init": learning_rate_init,
            "max_iter": max_iter,
        }
        parameters.update(kwargs)

        multilayer_perceptron_model = MLPRegressor(**parameters)

        super().__init__(model=multilayer_perceptron_model)
