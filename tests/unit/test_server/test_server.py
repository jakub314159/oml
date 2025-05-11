from build.lib.oml.classification.pipeline_creation import PipelineCreator
from oml.sql_connector import SQLConnector


def test_run_experiment(client):
    payload = {
        "file_name": "test_file_name",
        "target_column_name": "test_target_column_name",
        "learning_type": "test_learning_type",
        "test_metric": "test_test_metric",
    }
    response = client.post("/run_experiment", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"training_id", "input_payload"}


def test_get_status(client, mocker):
    payload = {"training_id": "training_id"}

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_record_from_table.return_value = {"status": "pending"}
    mocker.patch("oml.server.SQLConnector", return_value=mocked_connector)
    response = client.get("/get_status", json=payload)

    assert response.get_json()["status"] == "pending"
    assert response.status_code == 202


def test_get_pipeline(client, mocker):
    payload = {"file_name": "file_name"}

    mocked_pipeline_creator = mocker.MagicMock(PipelineCreator)
    mocked_pipeline_creator.from_pickle_to_bytes.return_value = bytes("pipeline", encoding="utf-8")
    mocker.patch("oml.server.PipelineCreator", return_value=mocked_pipeline_creator)

    response = client.get("/get_pipeline", json=payload)

    assert response == {"pipeline": {"file_name": "file_name"}}
