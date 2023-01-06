def test_models_equivalency(mock_regression_model_class):
    mock_class_1 = mock_regression_model_class()
    mock_class_2 = mock_regression_model_class()

    assert mock_class_1 == mock_class_2

    mock_class_1 = mock_regression_model_class(first_arg=4)
    mock_class_2 = mock_regression_model_class(first_arg=1)

    assert not mock_class_1 == mock_class_2

    mock_class_1 = mock_regression_model_class()
    mock_class_1.name = "mock Regression Model"
    mock_class_2 = mock_regression_model_class()

    assert not mock_class_1 == mock_class_2
