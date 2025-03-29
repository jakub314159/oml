from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from oml.pipeline_creation import PipelineCreator

_here = Path(__file__).parent


def test_regression_model_selector():
    data = pd.read_csv(_here.parent / "test_data" / "weatherHistory.csv")
    data = data.drop(columns=["Formatted Date"])
    data = data[:300]

    pipeline_creator = PipelineCreator(
        dataframe=data,
        target_column_name="Temperature (C)",
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
    assert pipeline_creator.pipeline_info[0]["score"] < 1
