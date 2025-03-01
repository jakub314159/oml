from sklearn.datasets import load_iris
import pandas as pd

from oml.regression.model_selection.model_selector import BaseModelSelector
from oml.regression.preprocessing.preprocessor import RegressionPreprocessor


def test_model_selector_on_iris_df():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

    preprocessor = RegressionPreprocessor(dataframe=iris_df, target_column_name="petal width (cm)")
    preprocessor.remove_invalid_data()
    preprocessor.data_target_split()

    selector = BaseModelSelector(preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test)
    best_models = selector.find_best_performing_models(n_best=3)

    assert len(best_models) == 3

    for model in best_models:
        assert 1 >= model["score"] >= 0
