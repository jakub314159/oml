import requests

post_payload = {
    "file_name": "ghosts_train",
    "target_column_name": "type",
    "learning_type": "classification",
    "test_metric": "recall",
}
# post_resp = requests.post(url="http://127.0.0.1:5000/run_experiment", json=post_payload)
# print(post_resp.status_code, post_resp.json())

# get_payload = {"training_id": "MjAyNTA1MTFfMTEwMDQyNzA2MzIxZ2hvc3RzX3RyYWlu"}
# get_resp = requests.get(url="http://127.0.0.1:5000/get_status", json=get_payload)
# print(get_resp.status_code, get_resp.json())

pipeline_payload = {"file_name": "ghosts_train"}
pipeline_resp = requests.get(url="http://127.0.0.1:5000/get_pipeline", json=pipeline_payload)
print(pipeline_resp.status_code, pipeline_resp)
