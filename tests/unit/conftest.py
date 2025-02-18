import pandas as pd
import pytest


@pytest.fixture
def mock_dataframe():
    return pd.DataFrame(
        {
            "x": [x for x in range(25)],
            "y": [x for x in range(25)],
            "target": [x for x in "abcabcacbabcacbacbacbacba"],
        }
    )


@pytest.fixture
def mock_train_test_data():
    X_train = pd.DataFrame({"x": [x for x in range(20)], "y": [x for x in range(20)]})
    X_test = pd.DataFrame({"x": [x for x in range(5)], "y": [x for x in range(5)]})
    y_train = pd.DataFrame(
        {
            "target": [x for x in range(20)],
        }
    )
    y_test = pd.DataFrame(
        {
            "target": [x for x in range(5)],
        }
    )
    return X_train, X_test, y_train, y_test