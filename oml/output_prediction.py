from pathlib import Path
from typing import Callable

import pandas as pd

from oml.pipeline_creation import PipelineCreator


def get_predict_function(file_path: Path) -> Callable:
    """
    Returns predict function.

    Parameters
    ----------
    file_path: Path
        The path in which the PipelineCreator is stored.

    Returns
    -------
    Callable: predict function
    """
    pipeline = PipelineCreator.from_pickle_to_pipeline(file_path)

    def predict_new_data(new_data: list | pd.DataFrame) -> list:
        """
        Function used to predict the target value for new data.

        Parameters
        ----------

        new_data: list | pd.Dataframe:
            Data without the target information.

        Returns
        -------
        pd.Dataframe: new_data as a dataframe with the predicted target column.
        """
        prediction_list = []

        if isinstance(new_data, pd.DataFrame):
            new_data = [pd.DataFrame([row], columns=new_data.columns) for row in new_data.to_numpy()]

        for sample_df in new_data:
            prediction = pipeline.predict(sample_df)[0]
            prediction_list.append(prediction)

        return prediction_list

    return predict_new_data
