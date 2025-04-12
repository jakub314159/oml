from flask import Flask, request
from pathlib import Path

import pandas as pd

from oml.pipeline_creation import PipelineCreator

app = Flask(__name__)


@app.route("/run_experiment", methods=["POST"])
def run_experiment():
    try:
        req = request.get_json()
        file_name = req.get("file_name")
        target_column_name = req.get("target_column_name")
        learning_type = req.get("learning_type")
        test_metric = req.get("test_metric")

        data = pd.read_csv(Path(__file__).parent / "data" / f"{file_name}.csv")

        pipeline_creator = PipelineCreator(
            dataframe=data, target_column_name=target_column_name, learning_type=learning_type, score_metric=test_metric
        )
        pipeline_creator.dataframe_preprocessing()
        pipeline_creator.prepare_model_selector()
        pipeline_creator.find_best_models(3)
        pipeline_creator.fit_best_pipelines(3)

        folder_path = Path(__file__).parent / "data" / f"{file_name}"
        folder_path.mkdir(exist_ok=True)

        pipeline_creator.save_pipeline_to_pickle_file(file_path=folder_path / "pipelines.pkl", index=0)
        status = "success"

    except Exception as e:
        status = "failed"

    return {"status": status}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
