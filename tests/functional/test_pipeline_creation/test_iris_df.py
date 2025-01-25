from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import pandas as pd

from oml.pipeline_creation import PipelineCreator


def test_pipeline_creator_on_iris_df():
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

    pipeline_creator = PipelineCreator(
        dataframe= iris_df,
        target_column_name= "species",
    )

    pipeline_creator.dataframe_preprocessing()
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_performing_models()
    pipeline_creator.get_n_best_scoring_models(test_metric="precision")
    pipeline = pipeline_creator.create_model_pipeline(index= 0)

    print(pipeline)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps["model"] == pipeline_creator.best_models[0]