from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from oml.pipeline_creation import PipelineCreator

def save_pipeline(file_path: Path) -> None:
    """
    Saves pipeline into desired file_path

    Parameters
    ----------
    file_path: Path
        The place where the pipeline will be stored.

    Returns
    -------
    None
    """
    iris = load_iris()

    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

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

save_pipeline(Path(__file__))