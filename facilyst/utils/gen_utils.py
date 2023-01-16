"""General utility functions."""
import importlib
from types import ModuleType
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import woodwork as ww

ww.config.set_option("numeric_categorical_threshold", 0.2)


def _get_subclasses(base_class: object) -> list:
    """Returns all subclasses to the base class passed.

    :param base_class:
    :type base_class: object
    :return: The list of child classes.
    :rtype: list
    """
    classes = base_class.__subclasses__()
    subclasses = []
    while classes:
        subclass = classes.pop()
        children_of_subclass = subclass.__subclasses__()
        if children_of_subclass:
            classes.extend(children_of_subclass)
        else:
            subclasses.append(subclass)
    return subclasses


error_str = (
    "{name} is not installed. Please install {name} using pip install {name} or install facilyst with extra "
    "dependencies using pip install facilyst[extra]."
)
import_errors_dict = {
    "catboost": error_str.format(name="catboost"),
    "xgboost": error_str.format(name="xgboost"),
    "torch": error_str.format(name="torch"),
    "transformers": error_str.format(name="transformers"),
    "sentencepiece": error_str.format(name="sentencepiece"),
    "keras_preprocessing": error_str.format(name="keras_preprocessing"),
}


def import_or_raise(library: str, error_msg: Optional[str] = None) -> ModuleType:
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
        msg = f"Missing extra dependency '{library}'. {error_msg}"
        raise ImportError(msg)


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
        "time series",
    ]:
        problem_type_ = "time series"
    else:
        raise ValueError("That problem type isn't recognized!")

    return problem_type_


def infer_problem_type(
    y: Union[pd.Series, np.ndarray], x: Optional[Union[pd.DataFrame, np.ndarray]] = None
) -> str:
    """Infers the most likely problem type based on the target data passed in.

    y (pd.Series, np.ndarray): The target data to be inferred.
    x (pd.DataFrame, np.ndarray): The features data. Only used in determining if the problem type is time series.
    returns (str): The inferred problem type
    """
    x, y = prepare_data(x, y, True)
    if not isinstance(y, (pd.Series, np.ndarray)):
        raise TypeError(
            "The target data must be of either pd.Series or np.ndarray type."
        )

    def _is_time_series(x):
        if x is None:
            return None
        if isinstance(x.index, pd.DatetimeIndex) and pd.infer_freq(x.index):
            return "time series"
        x_ww = x.ww.init()
        datetime_columns = x.ww.select(include="Datetime").columns
        if len(datetime_columns) >= 1:
            for col in datetime_columns:
                if pd.infer_freq(x_ww.ww[col]):
                    return "time series"
                continue
            return None
        else:
            return None

    y_ww = ww.init_series(y)
    if not y_ww.ww.schema.is_numeric:
        problem_type = "classification"
    else:
        problem_type = "regression"
        problem_type = _is_time_series(x) or problem_type

    return problem_type


def prepare_data(
    x: Union[pd.DataFrame, np.ndarray] = None,
    y: Union[pd.DataFrame, np.ndarray] = None,
    ww_initialize: bool = False,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Prepares the data for usage by models by converting to Pandas types and initializing through Woodwork.

    x (pd.DataFrame or np.ndarray): Training data. Optional.
    y (pd.Series or np.ndarray): Target data. Optional.
    ww_initialize (bool): Whether to initialize the resulting pandas objects through Woodwork.
    """
    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    if ww_initialize:
        if x is not None:
            x.ww.init()
        if y is not None:
            y = ww.init_series(y)

    return x, y
