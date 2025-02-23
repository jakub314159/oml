from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
import pandas as pd

from oml.classification.pipeline_creation import PipelineCreator


def test_pipeline_creator_on_iris_df(tmp_path):
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

    file_path = tmp_path / "pipeline_test.pkl"

    pipeline_creator = PipelineCreator(
        dataframe=iris_df,
        target_column_name="species",
    )

    pipeline_creator.dataframe_preprocessing()
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_models()
    pipeline_creator.fit_best_pipelines(test_metric="precision")
    pipeline = pipeline_creator.get_pipeline(index=0)
    pipeline_creator.save_pipeline_to_pickle_file(file_path, model_pipeline=pipeline)

    saved_pipeline = pipeline_creator.from_pickle_to_pipeline(file_path=file_path)

    assert isinstance(pipeline, Pipeline)
    assert file_path.exists(), "asserting that the path exists"
    assert isinstance(saved_pipeline, Pipeline)
