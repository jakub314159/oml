from flask import Flask, request, make_response, jsonify
from pathlib import Path

import pandas as pd
from threading import Thread
import base64
from datetime import datetime

from sklearn.pipeline import Pipeline

from oml.pipeline_creation import PipelineCreator
from oml.sql_connector import SQLConnector

_database_path = Path(__file__).parents[1] / "data" / "sqlite_base.db"


app = Flask(__name__)


def update_status(training_id, status):
    connector.update_record_in_table("status_database", "training_id", training_id, "status", status)


def run_worker(training_id, status, file_name, target_column_name, learning_type, test_metric):
    connector = SQLConnector(database_path=_database_path)

    try:
        connector.add_to_table(
            "status_database", [(training_id, status, file_name, target_column_name, learning_type, test_metric)]
        )

        data = pd.read_csv(Path(__file__).parent / "data" / f"{file_name}.csv")

        pipeline_creator = PipelineCreator(
            dataframe=data, target_column_name=target_column_name, learning_type=learning_type, score_metric=test_metric
        )

        status = "running"
        update_status(training_id, status)

        pipeline_creator.dataframe_preprocessing()
        pipeline_creator.prepare_model_selector()
        pipeline_creator.find_best_models(3)
        pipeline_creator.fit_best_pipelines(3)

        folder_path = Path(__file__).parent / "data" / f"{file_name}"
        folder_path.mkdir(exist_ok=True)

        pipeline_creator.save_pipeline_to_pickle_file(file_path=folder_path / "pipelines.pkl", index=0)

        status = "completed"
        update_status(training_id, status)

    except Exception as e:
        status = "failed"
        update_status(training_id, status)

        connector.add_to_table("error_database", (training_id, str(e)))


def generate_id(name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    combined = f"{timestamp}{name}"

    encoded = base64.b64encode(combined.encode("utf-8")).decode("utf-8")

    return encoded


@app.route("/run_experiment", methods=["POST"])
def run_experiment():
    try:
        req = request.get_json()
        file_name = req.get("file_name")
        target_column_name = req.get("target_column_name")
        learning_type = req.get("learning_type")
        test_metric = req.get("test_metric")

        training_id = generate_id(file_name)
        status = "pending"

        thread = Thread(
            target=run_worker, args=(training_id, status, file_name, target_column_name, learning_type, test_metric)
        )
        thread.start()

        training_id = file_name

        output_payload = {
            "training_id": training_id,
            "input_payload": req,
        }
        response = make_response(jsonify(output_payload), 202)

    except Exception as e:
        output_payload = {
            "error": str(e),
            "input_payload": request.get_json(),
        }
        response = make_response(jsonify(output_payload), 500)

    return response


@app.route("/get_status", methods=["GET"])
def get_status():
    req = request.get_json()
    training_id = req.get("training_id")

    connector = SQLConnector(database_path=_database_path)
    record = connector.get_record_from_table("status_database", "training_id", training_id)

    response = make_response(jsonify(record), 202)
    return response


@app.route("/get_pipeline", methods=["GET"])
def get_pipeline() -> Pipeline:
    req = request.get_json()
    file_name = req.get("file_name")

    file_path = Path(__file__).parent / "data" / f"{file_name}.csv"

    pipeline = PipelineCreator.from_pickle_to_bytes(file_path)
    #
    # data = bytes(pipeline, encoding="utf-8")

    response = make_response(pipeline)
    response.headers["Content-Type"] = "application/octet-stream"
    response.headers["Content-Disposition"] = "attachment; filename=example.txt"
    return response


if __name__ == "__main__":

    connector = SQLConnector(database_path=_database_path)
    if "status_database" not in connector.get_existing_tables():
        connector.create_table(
            "status_database",
            ("training_id", "status", "file_name", "target_column_name", "learning_type", "test_metric"),
        )
    if "error_database" not in connector.get_existing_tables():
        connector.create_table("error_database", ("training_id", "error"))

    app.run(host="0.0.0.0", port=5000)
