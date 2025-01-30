from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
import pandas as pd

from oml.pipeline_creation import PipelineCreator


def test_model_selector_on_breast_cancer():
    all_data = load_breast_cancer(as_frame=True)
    data = all_data["data"]
    target = all_data["target"]

    pipeline_creator = PipelineCreator(
        dataframe=pd.DataFrame,
        target_column_name="species",
    )

    pipeline_creator.preprocessor.data = data
    pipeline_creator.preprocessor.target = target
    pipeline_creator.preprocessor.train_test_split()
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_models()
    pipeline_creator.metric_test_and_fitting_best_models(test_metric="precision")
    pipeline = pipeline_creator.create_model_pipeline(index=0)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.steps[1] == ("encoder", pipeline_creator.preprocessor.encoder)
    assert pipeline.steps[2] == ("model", pipeline_creator.model_info[0]["model"])
