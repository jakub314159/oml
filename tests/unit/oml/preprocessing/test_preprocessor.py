import pytest
import pandas as pd
import numpy as np

from oml.preprocessing.preprocessor import Preprocessor

## Todo 2: improving tests adding comments  'given', adding prints after assert after comma.


@pytest.fixture()
def preprocessor():
    df = pd.DataFrame({"x": [x for x in range(10)], "y": [x for x in "aaabbbcccc"]})

    preprocessor = Preprocessor(dataframe=df, target_column_name="y")
    return preprocessor


class Test_Preprocessor:
    def test_remove_invalid_data(self, preprocessor):
        preprocessor.df["y"] = ["a", "a", np.nan, "b", "b", "b", np.nan, "c", "c", np.nan]
        preprocessor.remove_invalid_data()

        assert len(preprocessor.df["y"]) == 7

    def test_data_target_split(self, preprocessor):
        preprocessor.data_target_split()

        assert preprocessor.target.shape == (len(preprocessor.df), 1)
        assert not "y" in preprocessor.data.columns

    def test_get_dummies(self):
        data = {
            "x": ["set", "set", "set", "set", "set", "vir", "vir", "vir", "vir", "vir"],
            "y": [x for x in "aaabbbcccc"],
        }
        df = pd.DataFrame(data)
        preprocessor = Preprocessor(dataframe=df, target_column_name="y")
        preprocessor.data_target_split()

        preprocessor.get_dummies()

        assert set(preprocessor.data.columns) == {"x_set", "x_vir"}

    def test_map_target(self, preprocessor):
        preprocessor.data_target_split()
        preprocessor.map_target()

        expected_mapping = {"a": 0, "b": 1, "c": 2}
        assert preprocessor.target_mapping == expected_mapping

    def test_train_test_split(self, preprocessor):
        preprocessor.data_target_split()
        preprocessor.train_test_split(0.2)

        assert isinstance(preprocessor.X_train, pd.DataFrame), "X_train after train_test_split must be pd.Dataframe."
        assert preprocessor.X_train.shape == (8, 1)
        assert preprocessor.X_test.shape == (2, 1)
        assert preprocessor.y_train.shape == (8, 1)
        assert preprocessor.y_test.shape == (2, 1)

    def test_impute(self, preprocessor):
        preprocessor.df["x"] = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        preprocessor.data_target_split()
        preprocessor.train_test_split()
    
        preprocessor.impute()
    
        assert all(x for x in preprocessor.data["x"] if x != np.nan)
