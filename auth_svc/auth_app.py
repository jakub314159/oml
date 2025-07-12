from flask import Flask, request, make_response, jsonify, Response
from pathlib import Path
import secrets
import datetime

from oml.sql_connector import SQLConnector

_database_path = Path(__file__).parents[1] / "data" / "auth_database.db"

app = Flask(__name__)


@app.route("/register", methods=["POST"])
def register() -> Response:
    """
    Post method for registering a new user.

    Returns
    -------
    Response: response
    """
    req = request.get_json()
    username = req.get("username")
    password = req.get("password")

    connector = SQLConnector(database_path=_database_path)
    user_db = connector.get_table("user_database")

    if username in user_db["username"]:
        output_payload = {
            "error": "Username already exists.",
            "input_payload": request.get_json(),
        }
        response = make_response(jsonify(output_payload), 409)

    else:
        connector.add_to_table("user_database", [(username, password)])

        output_payload = {
            "username": username,
            "password": password,
        }

        response = make_response(jsonify(output_payload), 202)
    return response


def create_token() -> dict:
    """
    Creates a new token for an existing user.

    Returns
    -------
    dict: token and expiration date.
    """
    token = secrets.token_urlsafe(64)

    creation_date = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H")
    expiration_date = str(int(creation_date) + 2)

    return {"token": token, "expiration_date": expiration_date}


@app.route("/get_token", methods=["GET"])
def get_token() -> Response:
    """
    Allows user to create a new token.

    Returns
    -------
    Response: output_payload
    """
    req = request.get_json()
    username = req.get("username")
    password = req.get("password")

    connector = SQLConnector(database_path=_database_path)

    if password not in connector.get_record_from_table("user_database", "username", username)["password"]:
        output_payload = {
            "error": "Wrong password given.",
            "input_payload": request.get_json(),
        }

        response = make_response(jsonify(output_payload), 500)

        return response

    token_info = create_token()

    token = token_info["token"]
    expiration_date = token_info["expiration_date"]

    connector.add_to_table("token_database", [(token, username, expiration_date)])

    output_payload = {
        "token": token,
        "username": username,
        "expiration_date": expiration_date,
    }

    response = make_response(jsonify(output_payload), 202)
    return response


@app.route("/validate_token", methods=["GET"])
def validate_token() -> Response:
    """
    Checks if the token is valid.

    Returns
    -------
    Response: output_payload
    """
    req = request.get_json()
    token = req.get("token")

    connector = SQLConnector(database_path=_database_path)

    if token not in connector.get_table("token_database")["token"]:
        output_payload = {
            "case": "Wrong token given.",
            "is_valid": False,
        }

        response = make_response(jsonify(output_payload), 401)
        return response

    now = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H")

    if int(now) > int(connector.get_record_from_table("token_database", "token", token)["expiration_date"]):
        output_payload = {
            "case": "Token expired.",
            "is_valid": False,
        }

        response = make_response(jsonify(output_payload), 401)
        return response

    output_payload = {
        "case": "Token valid.",
        "is_valid": True,
    }

    response = make_response(jsonify(output_payload), 200)
    return response


if __name__ == "__main__":
    connector = SQLConnector(database_path=_database_path)

    if "user_database" not in connector.get_existing_tables():
        connector.create_table(
            "user_database",
            ("username", "password"),
        )

    if "token_database" not in connector.get_existing_tables():
        connector.create_table(
            "token_database",
            ("token", "username", "expiration_date"),
        )

    app.run(host="0.0.0.0", port=5000)
