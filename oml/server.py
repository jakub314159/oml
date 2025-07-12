from threading import Thread
from pathlib import Path
import base64

from datetime import datetime

import requests
from flask import Flask, request, make_response, jsonify, Response
import pandas as pd

from oml.pipeline_creation import PipelineCreator
from oml.sql_connector import SQLConnector
from oml import logger

_database_path = Path(__file__).parents[1] / "data" / "experiment_database.db"

app = Flask(__name__)

_auth_database_path = Path(__file__).parents[1] / "data" / "auth_database.db"


class FailedExperiment(Exception):
    """
    Exception used when the experiment fails.
    """


class WrongFileNameError(ValueError):
    """
    Error used when the user gives a wrong file_name to the experiment.
    """


class WrongLearningTypeError(ValueError):
    """
    Error used when the user gives a wrong learning_type to the experiment.
    """


class WrongTestMetricError(ValueError):
    """
    Error used when the user gives a wrong test_metric to the experiment.
    """


class WrongTargetColumnNameError(ValueError):
    """
    Error used when the user gives a wrong target_column_name to the experiment.
    """


def update_status(training_id: str, status: str, sql_connector) -> None:
    """
    Updates 'status' in the status_database

    Parameters
    ----------
    training_id: str
        The identification ticket for user.

    status: str
        The new status.

    sql_connector: SQLConnector
        The connector.

    Returns
    -------
    None
    """

    sql_connector.update_record_in_table("status_database", "training_id", training_id, "status", status)


def validate_parameters(file_name: str, target_column_name: str, learning_type: str, test_metric: str) -> None | dict:
    """
    Validates the parameters for the experiment.

    Parameters
    ----------
    file_name: str
        The name of the file.

    target_column_name: str
        The name of the target column.

    learning_type: str
        The learning type.

    test_metric: str
        The test metric.

    Returns
    -------
    None: if parameters are correct.
    dict: if there is an error.
    """
    file_path = Path(__file__).parents[1] / "data" / f"{file_name}.csv"

    if not file_path.exists():
        return {"error": "The file_name provided in the request is incorrect."}

    learning_types_used = ["classification", "regression"]

    if learning_type not in learning_types_used:
        return {"error": f"The learning type provided is incorrect. User must choose from {learning_types_used}"}

    metrics_used = ["accuracy", "recall", "f1", "precision", "rmse"]

    if test_metric not in metrics_used:
        return {
            "error": f"The test metric provided in the request is unsupported. User must choose from {metrics_used}"
        }

    dataframe = pd.read_csv(file_path)

    if target_column_name not in dataframe.columns:
        return {"error": "The target_column_name provided in the request doesn't exist in the dataframe."}

    return None


def run_worker(
    training_id: str, status: str, file_name: str, target_column_name: str, learning_type: str, test_metric: str
) -> None:
    """
    Starts the experiment.

    Parameters
    ----------
     training_id: str
        The identification ticket for user.

    status: str
        The new status.

    file_name: str
            Name of the file from which the Pipeline will learn.

    target_column_name: str
        The target for PipelineCreator

    learning_type: str
        Either 'classification' or 'regression'.

    test_metric: str
        The metric that will test the PipelineCreator

    Raises
    -------
    FailedExperiment: raised if wrong data is provided.

    Returns
    -------
    None
    """
    sql_connector = SQLConnector(database_path=_database_path)

    try:
        sql_connector.add_to_table(
            "status_database", [(training_id, status, file_name, target_column_name, learning_type, test_metric)]
        )

        dataframe = pd.read_csv(Path(__file__).parents[1] / "data" / f"{file_name}.csv")

        pipeline_creator = PipelineCreator(
            dataframe=dataframe,
            target_column_name=target_column_name,
            learning_type=learning_type,
            score_metric=test_metric,
        )

        status = "running"
        update_status(training_id, status, sql_connector)

        pipeline_creator.dataframe_preprocessing()
        pipeline_creator.prepare_model_selector()
        pipeline_creator.find_best_models(3)
        pipeline_creator.fit_best_pipelines(3)

        folder_path = Path(__file__).parents[1] / "data" / f"{file_name}_{training_id[5:9]}"
        folder_path.mkdir(exist_ok=True)

        pipeline_creator.save_pipeline_to_pickle_file(file_path=folder_path / "pipelines.pkl", index=0)

        status = "completed"
        update_status(training_id, status, sql_connector)

    except Exception as e:
        logger.warning(str(e), exc_info=True)

        status = "failed"
        update_status(training_id, status, sql_connector)

        sql_connector.add_to_table("error_database", [(training_id, str(e))])
        raise FailedExperiment(str(e)) from e


def generate_id(file_name: str) -> str:
    """
    Generates an id for the user based on the time and file_name coded in utf-8

    Parameters
    ----------
    file_name: str
        Name of the file from which the Pipeline will learn.

    Returns
    -------
    str: id
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    combined = f"{timestamp}{file_name}"

    encoded = base64.b64encode(combined.encode("utf-8")).decode("utf-8")

    return encoded


def validate_token(token: str) -> dict:
    """
    Validates the token that lets user use the app.

    Parameters
    ----------
    token: str
        The token that will be validated.

    Returns
    -------
    dict: payload with the token.
    """
    get_resp = requests.get(url=f"{"http://auth_svc:5000"}/validate_token", json={"token": token}, timeout=60).json()

    is_valid = get_resp["is_valid"]

    return is_valid


@app.route("/run_experiment", methods=["POST"])
def run_experiment() -> Response:
    """
    Post method used to start the experiment.

    Returns
    -------
    Response: The server response.
    """
    bearer_token = request.headers.get("Authorization")
    token = bearer_token.split(" ")[1]

    is_valid = validate_token(token)

    if not is_valid:
        return make_response(jsonify({"error": "Invalid token"}), 401)

    try:
        req = request.get_json()
        file_name = req.get("file_name")
        target_column_name = req.get("target_column_name")
        learning_type = req.get("learning_type")
        test_metric = req.get("test_metric")

        validation = validate_parameters(file_name, target_column_name, learning_type, test_metric)

        if validation is not None:
            return make_response(jsonify(validation), 401)

        training_id = generate_id(file_name)
        status = "pending"

        thread = Thread(
            target=run_worker, args=(training_id, status, file_name, target_column_name, learning_type, test_metric)
        )
        thread.start()

        output_payload = {
            "training_id": training_id,
            "input_payload": req,
        }
        response = make_response(jsonify(output_payload), 202)

    # pylint: disable=broad-exception-caught
    except Exception as e:
        output_payload = {
            "error": str(e),
            "input_payload": request.get_json(),
        }
        response = make_response(jsonify(output_payload), 500)

    return response


@app.route("/get_status", methods=["GET"])
def get_status() -> Response:
    """
    Get method used to get the status of the experiment.

    Returns
    -------
    Response: The server response.
    """
    bearer_token = request.headers.get("Authorization")
    token = bearer_token.split(" ")[1]

    is_valid = validate_token(token)

    if not is_valid:
        return make_response(jsonify({"error": "Invalid token"}), 401)

    req = request.get_json()
    training_id = req.get("training_id")

    sql_connector = SQLConnector(database_path=_database_path)
    record = sql_connector.get_record_from_table("status_database", "training_id", training_id)

    if record.get("status") == "failed":
        error_record = sql_connector.get_record_from_table("error_database", "training_id", training_id)
        response = make_response(jsonify(error_record), 202)

    else:
        response = make_response(jsonify(record), 202)
    return response


@app.route("/get_pipeline", methods=["GET"])
def get_pipeline() -> Response:
    """
    Get method used to get the Pipeline if the experiment is finished.

    Returns
    -------
    Response: The server response.
    """
    bearer_token = request.headers.get("Authorization")
    token = bearer_token.split(" ")[1]

    is_valid = validate_token(token)

    if not is_valid:
        return make_response(jsonify({"error": "Invalid token"}), 401)

    req = request.get_json()
    file_name = req.get("file_name")

    file_path = Path(__file__).parents[1] / "data" / f"{file_name}" / "pipelines.pkl"

    pipeline = PipelineCreator.from_pickle_to_bytes(file_path)

    response = make_response(pipeline)
    response.headers.set("Content-Type", "application/octet-stream")
    response.headers.set("Content-Disposition", "attachment", filename="pipeline.pkl")
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
