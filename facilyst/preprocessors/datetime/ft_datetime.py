"""Preprocessor creates datetime features using FeatureTools."""
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from featuretools import (
    EntitySet,
    IdentityFeature,
    calculate_feature_matrix,
    dfs,
)

from facilyst.preprocessors.preprocessor_base import PreprocessorBase


class FTDatetime(PreprocessorBase):
    """A preprocessor that adds FeatureTools computed datetime features.

    All datetime columns are feature engineered for the specified transform primitives and the original datetime columns are
    removed from the transformed dataframe. To ignore certain columns from being put through this feature engineering, pass
    their column names as a list to `cols_to_ignore`.

    :param index_column: The name of the column that represents the index. Defaults to "index" which refers to the dataframe's index.
    :type index_column: str, optional
    :param second: Whether to include the transform primitive to parse out the seconds from datetime columns. Defaults to True.
    :type second: bool, optional
    :param minute: Whether to include the transform primitive to parse out the minutes from datetime columns. Defaults to True.
    :type minute: bool, optional
    :param hour: Whether to include the transform primitive to parse out the hours from datetime columns. Defaults to True.
    :type hour: bool, optional
    :param day: Whether to include the transform primitive to parse out the day from datetime columns. Defaults to True.
    :type day: bool, optional
    :param week: Whether to include the transform primitive to parse out the week from datetime columns. Defaults to True.
    :type week: bool, optional
    :param month: Whether to include the transform primitive to parse out the month from datetime columns. Defaults to True.
    :type month: bool, optional
    :param year: Whether to include the transform primitive to parse out the year from datetime columns. Defaults to True.
    :type year: bool, optional
    :param weekday: Whether to include the transform primitive to parse out whether the datetime refers to a week day
    in datetime columns. Defaults to False.
    :type weekday: bool, optional
    :param is_weekend: Whether to include the transform primitive to parse out whether the datetime is during the weekend
    in datetime columns. Defaults to False.
    :type is_weekend: bool, optional
    :param is_federal_holiday: Whether to include the transform primitive to parse out whether the datetime is a federal holiday
    in datetime columns. Defaults to False.
    :type is_federal_holiday: bool, optional
    :param date_to_holiday: Whether to include the transform primitive to parse out what holiday the datetime represents
    in datetime columns. No holidays are represented as NaN. Defaults to False.
    :type date_to_holiday: bool, optional
    :param distance_to_holiday: Whether to include the transform primitive to parse out the time in days since New Year's Day
    in datetime columns. Defaults to False.
    :type distance_to_holiday: bool, optional
    :param cols_to_ignore: Columns to ignore for the feature engineering process.
    :type cols_to_ignore: list[str], optional
    """

    name: str = "FT Datetime"

    primary_type: str = "x"
    secondary_type: str = "featurize"
    tertiary_type: str = "datetime"

    hyperparameters: dict = {}

    def __init__(
        self,
        index_column: Optional[str] = None,
        second: bool = True,
        minute: bool = True,
        hour: bool = True,
        day: bool = True,
        week: bool = True,
        month: bool = True,
        year: bool = True,
        weekday: bool = False,
        is_weekend: bool = False,
        is_federal_holiday: bool = False,
        date_to_holiday: bool = False,
        distance_to_holiday: bool = False,
        cols_to_ignore: Optional[List[str]] = None,
        **kwargs,
    ):
        self.index_column = index_column or "index"
        self.features = None
        self.cols_to_ignore = cols_to_ignore or []
        if not isinstance(self.cols_to_ignore, list):
            raise TypeError("The parameter `cols_to_ignore` must be of type list.")
        parameters = {
            "trans_primitives": {
                "second": second,
                "minute": minute,
                "hour": hour,
                "day": day,
                "week": week,
                "month": month,
                "year": year,
                "weekday": weekday,
                "is_weekend": is_weekend,
                "is_federal_holiday": is_federal_holiday,
                "date_to_holiday": date_to_holiday,
                "distance_to_holiday": distance_to_holiday,
            }
        }

        parameters.update(kwargs)

        super().__init__(preprocessor=None, parameters=parameters)

    def _make_entity_set(self, x_ww):
        ft_es = EntitySet("DateTime_DataFrame")
        del x_ww.ww
        es = ft_es.add_dataframe(
            dataframe=x_ww,
            dataframe_name="my_df",
            index=self.index_column,
            make_index=True if self.index_column not in x_ww.columns else False,
        )
        return es

    def fit(
        self,
        x: Union[pd.DataFrame, np.ndarray],
        y: Any = None,
    ) -> PreprocessorBase:
        """Fits on the data using the preprocessor.

        :param x: The testing data for the preprocessor to fit on.
        :type x: pd.DataFrame or np.ndarray
        :param y: The testing data for the preprocessor to fit on. Ignored.
        :type y: pd.Series or np.array
        """
        x_ww = x[set(x.columns) - set(col for col in self.cols_to_ignore)].copy()
        x_ww.ww.init()
        x_ww = x_ww.ww.rename({col: str(col) for col in x_ww.columns})
        es = self._make_entity_set(x_ww)

        trans_prims = [
            prim for prim, used in self.parameters["trans_primitives"].items() if used
        ]
        fm, self.features = dfs(
            entityset=es,
            target_dataframe_name="my_df",
            features_only=False,
            max_depth=1,
            trans_primitives=trans_prims,
        )
        return self

    @staticmethod
    def _remove_no_unique_columns(feature_matrix, added_columns):
        for column in added_columns:
            if feature_matrix[column].nunique() == 1:
                feature_matrix.drop(column, axis=1, inplace=True)
        return feature_matrix

    def _separate_original_columns(self, x_ww):
        x_ww.ww.init()
        original_schema = x_ww.ww.schema
        if self.cols_to_ignore:
            ignored_columns_df = pd.concat(
                [x_ww.pop(x) for x in self.cols_to_ignore], axis=1
            )
        else:
            ignored_columns_df = None
        x_ww.ww.init()
        return x_ww, original_schema, ignored_columns_df

    def _combine_original_columns(
        self, x_ww, original_schema, ignored_columns_df, original_columns
    ):
        index_name = x_ww.index.name
        if ignored_columns_df is not None:
            x_ww = pd.concat([x_ww, ignored_columns_df], axis=1)
        x_ww.index.name = index_name or "index"

        original_columns = original_columns.union(set(self.cols_to_ignore))
        partial_schema = original_schema.get_subset_schema(subset_cols=original_columns)
        x_ww.ww.init(schema=partial_schema)
        return x_ww

    def transform(
        self,
        x: Union[pd.DataFrame],
        y: Any = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Transforms the data using the preprocessor.

        :param x: The testing data for the preprocessor to transform.
        :type x: pd.DataFrame
        :param y: The testing data for the preprocessor to transform. Ignored.
        :type y: pd.Series or np.array
        :return: The transformed data.
        :rtype tuple:
        """
        x_ww = x.copy()
        original_columns = x.columns
        x_ww, original_schema, ignored_columns_df = self._separate_original_columns(
            x_ww
        )
        x_ww = x_ww.ww.rename({col: str(col) for col in x_ww.columns})

        all_identity = all([isinstance(f, IdentityFeature) for f in self.features])
        if all_identity:
            x_ww = self._combine_original_columns(
                x_ww, original_schema, ignored_columns_df, original_columns
            )
            return x_ww, y

        es = self._make_entity_set(x_ww)
        feature_matrix = calculate_feature_matrix(features=self.features, entityset=es)

        added_columns = set(feature_matrix.columns) - set(x_ww.columns)
        feature_matrix = FTDatetime._remove_no_unique_columns(
            feature_matrix, added_columns
        )

        leftover_columns = set(x_ww.columns).intersection(set(feature_matrix.columns))

        x_ww = self._combine_original_columns(
            feature_matrix, original_schema, ignored_columns_df, leftover_columns
        )

        return x_ww, y
