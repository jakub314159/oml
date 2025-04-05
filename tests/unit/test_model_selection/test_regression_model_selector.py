from oml.regression.model_selection.model_selector import RegressionModelSelector


def test_regression_model_selector(mock_train_test_data):
    X_train, X_test, y_train, y_test = mock_train_test_data
    model_selector = RegressionModelSelector(X_train, X_test, y_train, y_test, "recall")

    assert (
        "linear_regression" in model_selector.models_mapping.keys()
        and "linear_regression" in model_selector.param_mapping.keys()
    )
