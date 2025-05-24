import requests

from sklearn.pipeline import Pipeline


class OMLClient:
    def __init__(self, url: None | str = None):
        """
        Class used to connecting to the server to run the experiment.

        Parameters
        ----------
        url : None | str, default=None
            <missing description>
        """
        self.url = url or "http://127.0.0.1:5000"
        self.training_id = None

    def run_experiment(self, file_name: str, target_column_name: str, learning_type: str, test_metric: str) -> dict:
        """
        Starts running the experiment and saves experiment data to database.

        Parameters
        ----------
        file_name: str
            Name of the file from which the Pipeline will learn.

        target_column_name: str
            The target for PipelineCreator

        learning_type: str
            Either 'classification' or 'regression'.

        test_metric: str
            The metric that will test the PipelineCreator

        Returns
        -------
        The response from the server.
        """
        post_payload = {
            "file_name": file_name,
            "target_column_name": target_column_name,
            "learning_type": learning_type,
            "test_metric": test_metric,
        }

        post_resp = requests.post(url=f"{self.url}/run_experiment", json=post_payload)

        self.training_id = post_resp.json().get("training_id")

        print(f"Status code {post_resp.status_code}")

        return post_resp.json()

    def get_status(self) -> dict:
        """
        Gives user status of their experiment.

        Returns
        -------
        The response from the server with the status.
        """
        get_payload = {"training_id": self.training_id}
        get_resp = requests.get(url=f"{self.url}/get_status", json=get_payload)

        print(f"Status code {get_resp.status_code}")

        return get_resp.json()

    def get_pipeline(self, file_name: str) -> Pipeline:
        """
        Returns the Pipeline if the experiment is done

        Parameters
        ----------
        file_name: str
            Name of the file from which the Pipeline has learned.

        Returns
        -------
        The response from the server with the Pipeline.
        """
        import pickle
        from io import BytesIO

        pipeline_payload = {"file_name": file_name}
        pipeline_resp = requests.get(url=f"{self.url}/get_pipeline", json=pipeline_payload, stream=True)

        print(f"Status code {pipeline_resp.status_code}")

        return pickle.load(BytesIO(pipeline_resp.content))


if __name__ == "__main__":
    import time

    client = OMLClient()
    print(client.run_experiment("ghosts_train", "type", "classification", "recall"))
    print(client.training_id)

    time.sleep(2)

    print(client.get_status())

    pipeline = client.get_pipeline("ghosts_train")

    print(pipeline)
