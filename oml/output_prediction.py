import random

import pandas as pd

from oml.pipeline_creation import PipelineCreator


def predict_new_data(dataframe: pd.DataFrame, target_column_name: str, new_data: list | pd.DataFrame):
    """
    Function used to predict the target value for new data.

    Parameters
    ----------
    dataframe: pd.Dataframe
        The whole dataframe containing data and target.

    target_column_name: str
        The name of the column that is the target.

    new_data: list | pd.Dataframe:
        Data without the target information.

    Returns
    -------
    pd.Dataframe: new_data as a dataframe with the predicted target column.
    """
    pipeline_creator = PipelineCreator(dataframe, target_column_name)

    pipeline_creator.dataframe_preprocessing()
    pipeline_creator.prepare_model_selector()
    pipeline_creator.find_best_models(1)
    pipeline_creator.fit_best_pipelines(n_best=1)

    pipeline = pipeline_creator.get_pipeline(0)

    prediction_list = []

    if isinstance(new_data, pd.DataFrame):
        new_data = [pd.DataFrame([row], columns=new_data.columns) for row in new_data.to_numpy()]

    for sample_df in new_data:
        prediction = pipeline.predict(sample_df)[0]

        mapping = {y: x for x, y in pipeline_creator.preprocessor.target_mapping.items()}
        prediction = mapping[prediction]

        prediction_list.append(prediction)

    merged_data = pd.concat(new_data, ignore_index=True)
    merged_data.to_dict()
    merged_data[target_column_name] = prediction_list

    return pd.DataFrame(merged_data)


if __name__ == "__main__":
    data = {
        "bones": [round(random.uniform(0, 10), 2) for _ in range(21)],
        "limbs": [round(random.uniform(0, 10), 2) for _ in range(21)],
        "species": [random.choice(["human", "tiki", "vampire"]) for _ in range(21)],
    }

    df = pd.DataFrame(data)

    test_data = {
        "bones": [round(random.uniform(0, 10), 2) for _ in range(10)],
        "limbs": [round(random.uniform(0, 10), 2) for _ in range(10)],
    }

    raw_data_df = pd.DataFrame(test_data)

    print(predict_new_data(df, "species", raw_data_df))
