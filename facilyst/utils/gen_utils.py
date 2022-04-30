"""General utility functions."""
import importlib
from types import ModuleType


def _get_subclasses(base_class: object) -> list:
    """Returns all subclasses to the base class passed.

    :param base_class:
    :type base_class: object
    :return: The list of child classes.
    :rtype: list
    """
    classes_to_check = base_class.__subclasses__()

    subclasses = []

    while classes_to_check:
        subclass = classes_to_check.pop()
        subclasses.append(subclass)

    return subclasses


error_str = (
    "{name} is not installed. Please install {name} using pip install {name} or install facilyst with extra "
    "dependencies using pip install facilyst[extra]."
)
import_errors = {
    "catboost": error_str.format("catboost"),
    "xgboost": error_str.format("xgboost"),
    "torch": error_str.format("torch"),
    "hyperopt": error_str.format("hyperopt"),
    "transformers": error_str.format("transformers"),
    "sentencepiece": error_str.format("sentencepiece"),
    "keras": error_str.format("keras"),
    "Keras-Preprocessing": error_str.format("Keras-Preprocessing"),
}


def import_or_raise(library: str, error_msg: bool = None) -> ModuleType:
    """Import the requested library.

    :param library: The name of the library.
    :type library: str
    :param error_msg: The error message to return if the import fails.
    :type error_msg: str
    :return: The imported library.
    :rtype: ModuleType
    :raises ImportError: If the library is not installed.
    :raises Exception: A general exception to not being able to import the library.
    """
    try:
        return importlib.import_module(library)
    except ImportError:
        if error_msg is None:
            error_msg = ""
        msg = f"Missing optional dependency '{library}'. Please use pip to install {library}. {error_msg}"
        raise ImportError(msg)
    except Exception as exception_msg:
        msg = f"An exception occurred while trying to import `{library}`: {str(exception_msg)}"
        raise Exception(msg)


def handle_problem_type(problem_type: str) -> str:
    """Handles the problem type passed to be returned in a consistent way.

    :param problem_type: The problem type to match.
    :type problem_type: str
    :return: The standardized problem type.
    :rtype: str
    """
    if problem_type.lower() in ["regression", "regressor"]:
        problem_type_ = "regression"
    elif problem_type.lower() in ["classification", "classifier"]:
        problem_type_ = "classification"
    elif problem_type.lower() in ["binary"]:
        problem_type_ = "binary"
    elif problem_type.lower() in ["multiclass", "multi", "multi class"]:
        problem_type_ = "multiclass"
    elif problem_type.lower() in [
        "time series regression",
        "timeseries regression",
        "ts regression",
    ]:
        problem_type_ = "time series regression"
    else:
        raise ValueError("That problem type isn't recognized!")

    return problem_type_
