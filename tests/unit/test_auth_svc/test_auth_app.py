from auth_svc.auth_app import create_token
from oml.sql_connector import SQLConnector


def test_register(client, mocker):
    payload = {
        "username": "username",
        "password": "password",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_table.return_value = {"username": ["abc"], "password": ["<PASSWORD>"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.post("/register", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"username", "password"}


def test_register_username_taken(client, mocker):
    payload = {
        "username": "username",
        "password": "password",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_table.return_value = {"username": ["username"], "password": ["<PASSWORD>"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.post("/register", json=payload)
    assert response.status_code == 409
    data = response.get_json()
    assert set(data.keys()) == {"error", "input_payload"}


def test_create_token():
    token = create_token()

    assert set(token.keys()) == {"token", "expiration_date"}


def test_get_token_wrong_password(client, mocker):
    payload = {
        "username": "username",
        "password": "password",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_record_from_table.return_value = {"password": ["<PASSWORD>"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/get_token", json=payload)
    assert response.status_code == 500
    data = response.get_json()
    assert set(data.keys()) == {"error", "input_payload"}


def test_get_token_update_token(client, mocker):
    payload = {
        "username": "username",
        "password": "password",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_record_from_table.return_value = {"password": ["password"]}
    mocked_connector.get_table.return_value = {"username": ["username"], "password": ["password"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/get_token", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"username", "token", "expiration_date"}


def test_get_token_update_value(client, mocker):
    payload = {
        "username": "username",
        "password": "password",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_record_from_table.return_value = {"password": ["password"]}
    mocked_connector.get_table.return_value = {"username": ["username"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/get_token", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"username", "token", "expiration_date"}


def test_get_token(client, mocker):
    payload = {
        "username": "username",
        "password": "password",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_record_from_table.return_value = {"password": ["password"]}
    mocked_connector.get_table.return_value = {"username": [], "password": ["password"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/get_token", json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"username", "token", "expiration_date"}


def test_validate_token_wrong_token(client, mocker):
    payload = {
        "token": "token",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_table.return_value = {"token": ["t"]}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/validate_token", json=payload)
    assert response.status_code == 401
    data = response.get_json()
    assert set(data.values()) == {"Wrong token given.", False}


def test_validate_token_expired_token(client, mocker):
    payload = {
        "token": "token",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_table.return_value = {"token": ["token"]}
    mocked_connector.get_record_from_table.return_value = {"expiration_date": "1"}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/validate_token", json=payload)
    assert response.status_code == 401
    data = response.get_json()
    assert set(data.values()) == {"Token expired.", False}


def test_validate_token_valid_token(client, mocker):
    payload = {
        "token": "token",
    }

    mocked_connector = mocker.MagicMock(SQLConnector)
    mocked_connector.get_table.return_value = {"token": ["token"]}
    mocked_connector.get_record_from_table.return_value = {"expiration_date": "11111111111"}
    mocker.patch("auth_svc.auth_app.SQLConnector", return_value=mocked_connector)

    response = client.get("/validate_token", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert set(data.values()) == {"Token valid.", True}
