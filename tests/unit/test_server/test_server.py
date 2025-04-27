def test_hello(client):
    payload = {
        "file_name": "test_file_name",
        "target_column_name": "test_target_column_name",
        "learning_type": "test_learning_type",
        "test_metric": "test_test_metric",
    }
    response = client.post('/run_experiment', json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert set(data.keys()) == {"training_id", "input_payload"}


def test_submit_job(client):
    payload = {'job_id': 123}
    response = client.post('/submit', json=payload)
    assert response.status_code == 202
    data = response.get_json()
    assert data['message'] == 'Job started'
    assert data['job_id'] == 123