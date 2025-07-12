import pytest
from auth_svc.auth_app import app  # import your Flask app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
