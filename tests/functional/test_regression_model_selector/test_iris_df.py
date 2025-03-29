from sklearn.datasets import load_iris
import pandas as pd

from oml.regression.model_selection.model_selector import RegressionModelSelector
from oml.machine_learning_components.base_preprocessor import BasePreprocessor


def test_model_selector_on_iris_df():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))
    iris_df = iris_df.drop(columns=["species"])

    preprocessor = BasePreprocessor(dataframe=iris_df, target_column_name="sepal width (cm)")
    preprocessor.remove_invalid_data()
    preprocessor.data_target_split()
    preprocessor.train_test_split()
    preprocessor.transform()

    selector = RegressionModelSelector(
        preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test, "rmse"
    )
    best_models = selector.find_best_performing_models(n_best=2)

    assert len(best_models) == 2
    assert "linear_regression" or "decision_tree_regression" in [x["model_name"] for x in best_models]
