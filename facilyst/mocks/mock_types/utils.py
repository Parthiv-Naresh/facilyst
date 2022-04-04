import numpy as np
import pandas as pd
from faker import Faker


def handle_mock_and_library_type(mock_type="features", library="pandas"):
    if mock_type.lower() in ["df", "dataframe", "features", "x"]:
        mock_type_ = "features"
    elif mock_type.lower() in ["dates", "date"]:
        mock_type_ = "dates"
    elif mock_type.lower() in ["waves", "wave", "sine", "sin", "cosine", "cos"]:
        mock_type_ = "wave"
    else:
        mock_type_ = "features"

    if library.lower() in ["pd", "pandas", "df", "dataframe", "series"]:
        library_ = "pandas"
    elif library.lower() in ["np", "numpy", "array", "ndarray"]:
        library_ = "numpy"
    else:
        library_ = "pandas"

    return mock_type_, library_


def mock_features_dtypes(num_rows=100):
    """
    Internal function that returns the default full dataset.

    :param num_rows: The number of observations in the final dataset. Defaults to 100.
    :return: The dataset with all columns included.
    """
    fake = Faker()

    def _remove_x_from_number(phone):
        if "x" in phone:
            phone = phone[: phone.find("x")]
        return phone

    phone_numbers = pd.Series([fake.phone_number() for _ in range(num_rows)])
    phone_numbers = phone_numbers.apply(_remove_x_from_number)

    def _remove_newline_from_address(address):
        address = address.replace("\n", ", ")
        return address

    addresses = pd.Series([fake.address() for _ in range(num_rows)])
    addresses = addresses.apply(_remove_newline_from_address)

    dtypes_dict = {
        "ints": [i for i in range(-num_rows // 2, num_rows // 2)],
        "rand_ints": np.random.choice([i for i in range(-5, 5)], num_rows),
        "floats": [float(i) for i in range(-num_rows // 2, num_rows // 2)],
        "rand_floats": np.random.uniform(low=-5.0, high=5.0, size=num_rows),
        "booleans": np.random.choice([True, False], num_rows),
        "categoricals": np.random.choice(
            ["First", "Second", "Third", "Fourth"], num_rows
        ),
        "dates": pd.date_range("1/1/2001", periods=num_rows),
        "texts": [
            f"My children are miserable failures, all {i} of them!"
            for i in range(num_rows)
        ],
        "ints_nullable": np.random.choice(
            [i for i in range(-10 // 2, 10 // 2)] + [pd.NA], num_rows
        ),
        "floats_nullable": np.random.choice(
            np.append([float(i) for i in range(-5, 5)], pd.NA), num_rows
        ),
        "booleans_nullable": np.random.choice([True, False, None], num_rows),
        "full_names": pd.Series([fake.name() for _ in range(num_rows)]),
        "phone_numbers": phone_numbers,
        "addresses": addresses,
        "countries": pd.Series([fake.country() for _ in range(num_rows)]),
        "email_addresses": pd.Series(
            [fake.ascii_free_email() for _ in range(num_rows)]
        ),
        "urls": pd.Series([fake.url() for _ in range(num_rows)]),
        "currencies": pd.Series([fake.pricetag() for _ in range(num_rows)]),
        "file_paths": pd.Series([fake.file_path(depth=3) for _ in range(num_rows)]),
        "ipv4": pd.Series([fake.ipv4() for _ in range(num_rows)]),
        "ipv6": pd.Series([fake.ipv6() for _ in range(num_rows)]),
        "lat_longs": pd.Series([fake.latlng() for _ in range(num_rows)]),
    }
    return dtypes_dict
