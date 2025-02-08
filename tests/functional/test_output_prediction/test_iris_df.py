from sklearn.datasets import load_iris
import pandas as pd
from pathlib import Path

from oml.output_prediction import get_predict_function


def test_scoring():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

    predict_func = get_predict_function(Path(__file__).parents[1] / "test_data" / "test_pipeline.pkl")
    prediction = predict_func(new_data=iris_df.head())

    assert len(prediction) == 5
    assert set(prediction) == {"setosa"}
