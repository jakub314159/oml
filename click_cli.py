from pathlib import Path

import click
import pandas as pd

from oml.pipeline_creation import PipelineCreator


@click.command()
@click.argument("file_name")
@click.argument("target_column_name")
@click.argument("learning_type")
@click.argument("test_metric")
def run_experiment(file_name, target_column_name, learning_type, test_metric):
    data = pd.read_csv(Path(__file__).parent / "data" / f"{file_name}.csv")

    pipeline_creator = PipelineCreator(
        dataframe=data, target_column_name=target_column_name, learning_type=learning_type, score_metric=test_metric
    )
    pipeline_creator.dataframe_preprocessing()
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_models(3)
    pipeline_creator.fit_best_pipelines(3)
    pipeline_creator.save_pipeline_to_pickle_file(
        file_path=Path(__file__).parent / "data" / f"{file_name}" / "pipelines.pkl", index=0
    )


if __name__ == "__main__":
    run_experiment()
