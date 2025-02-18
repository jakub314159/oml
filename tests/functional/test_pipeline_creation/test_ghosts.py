from pathlib import Path

from sklearn.pipeline import Pipeline
import pandas as pd

from oml.pipeline_creation import PipelineCreator

_here = Path(__file__).parent


def test_model_selector_on_ghosts():
    data = pd.read_csv(_here.parent / "test_data" / "ghosts_train.csv")
    data = data.drop(columns=["id"])

    pipeline_creator = PipelineCreator(dataframe=data, target_column_name="type", strategy="mean")
    pipeline_creator.dataframe_preprocessing(train_test_split_size=0.25)
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_models()
    pipeline_creator.fit_best_pipelines(test_metric="precision")
    pipeline = pipeline_creator.get_pipeline(index=0)

    assert isinstance(pipeline, Pipeline)
    pred_df = pd.DataFrame(
        data={
            "bone_length": [0.64],
            "rotting_flesh": [0.532],
            "hair_length": [0.74],
            "has_soul": [0.89],
            "color": "white",
        }
    )
    assert pipeline.predict(pred_df)
