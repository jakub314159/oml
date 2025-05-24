from oml.sql_connector import SQLConnector


def test_run_experiment(client):
    payload = {
        "file_name": "ghosts_train",
        "target_column_name": "type",
        "learning_type": "classification",
        "test_metric": "recall",
        "skip": 0,
    }
    response = client.post("/run_experiment", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"training_id", "input_payload"}


def test_run_experiment_wrong_file_name(client):
    payload = {
        "file_name": "test_file_name",
        "target_column_name": "type",
        "learning_type": "classification",
        "test_metric": "recall",
    }
    response = client.post("/run_experiment", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data.get("error") == "The file_name provided in the request is incorrect."


def test_run_experiment_wrong_learning_type(client):
    payload = {
        "file_name": "ghosts_train",
        "target_column_name": "type",
        "learning_type": "test_learning_type",
        "test_metric": "recall",
        "skip": 1,
    }
    response = client.post("/run_experiment", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert (
        data.get("error")
        == "The learning type provided in the request is incorrect. User must choose from ['classification', 'regression']"
    )


def test_run_experiment_wrong_test_metric(client):
    payload = {
        "file_name": "ghosts_train",
        "target_column_name": "type",
        "learning_type": "classification",
        "test_metric": "test_test_metric",
        "skip": 2,
    }
    response = client.post("/run_experiment", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert (
        data.get("error")
        == "The test metric provided in the request is unsupported. User must choose from ['accuracy', 'recall', 'f1', 'precision', 'rmse']"
    )


def test_run_experiment_wrong_target_column_name(client):
    payload = {
        "file_name": "ghosts_train",
        "target_column_name": "test_target_column_name",
        "learning_type": "classification",
        "test_metric": "recall",
        "skip": 3,
    }
    response = client.post("/run_experiment", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data.get("error") == "The target_column_name provided in the request doesn't exist in the dataframe."


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

    expected_value = bytes("pipeline", encoding="utf-8")

    mocker.patch("oml.server.PipelineCreator.from_pickle_to_bytes", return_value=expected_value)

    response = client.get("/get_pipeline", json=payload)

    assert response.data == expected_value
