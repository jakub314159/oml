import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer

from oml import logger
from oml.classification.preprocessing import Preprocessor


@pytest.fixture
def preprocessor(mock_dataframe):
    return Preprocessor(dataframe=mock_dataframe, target_column_name="target")


@pytest.fixture
def split_preprocessor(preprocessor, data, target):
    preprocessor.data = data
    preprocessor.target = target

    return preprocessor


@pytest.fixture
def data():
    return pd.DataFrame({"x": [x for x in range(25)], "y": [y for y in range(25)]})


@pytest.fixture
def target():
    return pd.DataFrame({"target": [x for x in "abcabcacbabcacbacbacbacba"]})


class TestPreprocessor:
    def test_remove_invalid_data(self, preprocessor, mocker):
        mocker.patch.object(logger, "info")
        preprocessor.remove_invalid_data()

        assert len(preprocessor.df) == 25
        logger.info.assert_called_with("invalid data was removed.")

    def test_remove_invalid_data_value_error(self, preprocessor):
        preprocessor.df = preprocessor.df.head(5)

        with pytest.raises(ValueError) as err:
            preprocessor.remove_invalid_data()
        assert str(err.value) == "Dataframe needs more than 20 valid rows."

    def test_data_target_split(self, preprocessor, mocker):
        mocker.patch.object(logger, "info")

        preprocessor.data_target_split()

        assert preprocessor.target.shape == (25, 1)
        assert "target" not in preprocessor.data.columns
        logger.info.assert_called_with("Splitting done.")

    def test_train_test_split(self, split_preprocessor, mocker):
        mocker.patch.object(logger, "info")

        split_preprocessor.train_test_split()

        assert split_preprocessor.X_train.shape == (20, 2)
        assert split_preprocessor.y_test.shape == (5, 1)

    def test_transform(self, split_preprocessor, data, mocker):
        mocker.patch.object(logger, "info")
        split_preprocessor.X_train = data[:20]

        transformer = split_preprocessor.transform()

        assert isinstance(transformer, ColumnTransformer)
