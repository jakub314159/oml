from pytest import fixture
import pandas as pd
import numpy as np


from oml.model_selection.model_selector import ModelSelector
from oml.preprocessing.preprocessor import Preprocessor

@fixture()
def selector():
    df = pd.DataFrame({"x": [x for x in range(10)], "y": [x for x in "aaabbbcccc"]})
    
    preprocessor = Preprocessor(dataframe=df, target_column_name="y")
    X_train, X_test, y_train, y_test = preprocessor.process()

    selector = ModelSelector(X_train, X_test, y_train, y_test)

    return selector


class TestModelSelector:
    def test_get_all_model_scores(self, selector):
        selector.param_mapping["KNN"] = {"n_neighbours": [1, 12]}
        selector.get_all_model_scores(model_name="KNN")

        assert len(selector.model_scores) == 2
        assert selector.model_scores[0] == "KNN"
        assert selector.model_scores[0] == 1