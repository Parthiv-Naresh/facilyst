import numpy as np
import pandas as pd
import pytest
from woodwork.logical_types import Datetime

from facilyst.preprocessors import FTDatetime


def test_ft_datetime_class_variables():
    assert FTDatetime.name == "FT Datetime"
    assert FTDatetime.primary_type == "x"
    assert FTDatetime.secondary_type == "featurize"
    assert FTDatetime.tertiary_type == "datetime"
    assert FTDatetime.hyperparameters == {}


def test_ft_datetime_cols_to_ignore_type_errors():
    with pytest.raises(
        TypeError, match="The parameter `cols_to_ignore` must be of type list."
    ):
        FTDatetime(cols_to_ignore={"column": "ignore"})


@pytest.mark.parametrize(
    "index_column, second, minute, hour, day, week, month, year, weekday, is_weekend, "
    "is_federal_holiday, date_to_holiday, distance_to_holiday, cols_to_ignore",
    [
        (
            None,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            None,
        ),
        (
            "Index_col",
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            None,
        ),
        (
            None,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            ["Dates_1"],
        ),
        (
            "Index_col",
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            ["Ints", "Year"],
        ),
        (
            None,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            ["Dates_1", "Dates_2"],
        ),
        (
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            ["Dates_1", "Dates_2"],
        ),
        (
            None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            None,
        ),
    ],
)
def test_ft_datetime(
    index_column,
    second,
    minute,
    hour,
    day,
    week,
    month,
    year,
    weekday,
    is_weekend,
    is_federal_holiday,
    date_to_holiday,
    distance_to_holiday,
    cols_to_ignore,
):
    kw_args = locals()

    index_column_ = kw_args.pop("index_column")
    cols_to_ignore_ = kw_args.pop("cols_to_ignore")
    trans_primitives_ = kw_args

    df = pd.DataFrame()
    df["Index_col"] = [i for i in range(100)]
    df["Ints"] = np.random.choice([1, 2, 3, 4], 100)
    df["Year"] = np.random.choice([1992, 1995, 2003, 2007], 100)
    df["Month"] = np.random.choice([1, 1, 4, 7, 9, 12], 100)
    df["Day"] = np.random.choice([1, 5, 7, 12, 25, 30], 100)
    df["Dates_1"] = pd.date_range("1/1/21", freq="3D", periods=100)
    df["Dates_2"] = pd.date_range("3/5/1996", freq="4H", periods=100)

    y = pd.Series([i for i in range(100)])

    df_test = df.copy()
    df_test.ww.init()

    ft_dt = FTDatetime(
        index_column=index_column_, cols_to_ignore=cols_to_ignore_, **trans_primitives_
    )
    ft_dt.fit(df, y)
    output_x, output_y = ft_dt.transform(df, y)

    pd.testing.assert_series_equal(y, output_y)
    assert len(output_x) == 100
    if index_column:
        assert output_x.index.name == index_column_
    else:
        assert output_x.index.name == "index"

    non_dt_columns = df_test.ww.schema._filter_cols(exclude=Datetime)
    for non_dt in non_dt_columns:
        if non_dt == index_column_:
            assert non_dt not in output_x.columns
        else:
            assert non_dt in output_x.columns

    dt_columns = df_test.ww.schema._filter_cols(include=Datetime)
    if any(val for _, val in kw_args.items()):
        if cols_to_ignore_:
            for dt_column in dt_columns:
                if dt_column in cols_to_ignore_:
                    assert dt_column in output_x.columns
                else:
                    assert dt_column not in output_x.columns
        else:
            assert set(dt_columns) - set(output_x.columns) == set(dt_columns)
