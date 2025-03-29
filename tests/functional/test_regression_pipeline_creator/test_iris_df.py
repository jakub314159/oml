from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

from oml.pipeline_creation import PipelineCreator

_here = Path(__file__).parent


def test_regression_model_selector():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

    pipeline_creator = PipelineCreator(
        dataframe=iris_df,
        target_column_name="sepal width (cm)",
        strategy="mean",
        learning_type="regression",
        score_metric="rmse",
    )

    pipeline_creator.dataframe_preprocessing(train_test_split_size=0.25)
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_models()
    pipeline_creator.fit_best_pipelines()
    pipeline = pipeline_creator.get_pipeline(index=0)

    assert isinstance(pipeline, Pipeline)
    assert pipeline_creator.pipeline_info[0]["score"] < 0.4
