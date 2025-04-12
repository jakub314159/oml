import requests

payload = {
    "file_name": "ghosts_train",
    "target_column_name": "type",
    "learning_type": "classification",
    "test_metric": "recall",
}

resp = requests.post(url="http://127.0.0.1:5000/run_experiment", json=payload)
print(resp.json())
