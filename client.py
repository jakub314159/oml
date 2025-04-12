import requests


# post_payload = {
#     "file_name": "ghosts_train",
#     "target_column_name": "type",
#     "learning_type": "classification",
#     "test_metric": "recall",
# }
# post_resp = requests.post(url="http://127.0.0.1:5000/run_experiment", json=post_payload)
# print(post_resp.json())
#

get_payload = {"file_name": "ghosts_train"}
get_resp = requests.get(url="http://0.0.0.0:5000/get_status", json=get_payload)
print(get_resp.json())


#
# post_payload = {
#     "file_name": "weatherHistory",
#     "target_column_name": "Apparent Temperature (C)",
#     "learning_type": "regression",
#     "test_metric": "rmse",
# }
# post_resp = requests.post(url="http://127.0.0.1:5000/run_experiment", json=post_payload)
# print(post_resp.json())
#
#
# get_payload = {"file_name": "weatherHistory"}
# get_resp = requests.get(url="http://127.0.0.1:5000/get_status", json=get_payload)
# print(get_resp.json())
