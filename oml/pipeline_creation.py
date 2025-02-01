from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

from oml.preprocessing.preprocessor import Preprocessor, SimpleImputer
from oml.model_selection.model_selector import ModelSelector
from oml.utils.utils_finding_best_models import utils_find_best_performing_models


class PipelineCreator:
    """
    Class used to create pipelines with chosen models.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe that will be processed.

    target_column_name: str
        The name of the target column.

    strategy: str
        Strategy to the imputer.
    """

    def __init__(self, dataframe: pd.DataFrame, target_column_name: str, strategy: str = "median"):
        self.preprocessor = Preprocessor(dataframe, target_column_name, strategy)
        self.imputer: SimpleImputer | None = None
        self.encoder: OneHotEncoder | None = None
        self.selector: ModelSelector | None = None

        self.transformer: ColumnTransformer | None = None

        self.best_models = {}
        self.model_info = []

    def dataframe_preprocessing(self, train_test_split_size: float = 0.2) -> None:
        """
        Does preprocessing on the dataframe.

        Parameters
        ----------
        train_test_split_size: float
            The size of the test df.
        Returns
        -------
        None
        """
        self.preprocessor.remove_invalid_data()
        self.preprocessor.data_target_split()
        # self.preprocessor.get_dummies()
        self.preprocessor.map_target()
        self.preprocessor.train_test_split(test_size=train_test_split_size)
        # self.preprocessor.impute()
        # self.imputer = self.preprocessor.imputer

        self.transformer = self.preprocessor.transform()

    def prepare_model_selector(self) -> None:
        """
        Prepares model selector for classification.

        Returns
        -------
        None
        """
        self.selector = ModelSelector(
            self.preprocessor.X_train, self.preprocessor.X_test, self.preprocessor.y_train, self.preprocessor.y_test
        )

    def add_models_to_selector(
        self, model_name: str, model_class, parameter_name: str = None, parameter_vals: list[int] = None
    ) -> None:
        """
        Adds models to the model selector.

        Parameters
        ----------
        model_name: str
            The name of the model you want to add.
        model_class: Any
            The class of the model you want to add (for example: LogisticRegression)
        parameter_name: str | None
            If you want to add params this is the name of the param.
        parameter_vals: list(int) | None
            If you want to add params this is the list of the values.
        Returns
        -------
        None
        """
        self.selector.models_mapping[model_name] = model_class
        self.selector.model_list = list(self.selector.models_mapping.keys())

        self.selector.param_mapping[model_name] = {parameter_name: parameter_vals}

    def add_parameters_to_selector_model(self, model_name, parameter_name: str, parameter_vals: list[int]) -> None:
        """
        Adds params to an to the selector param mapping.

        Parameters
        ----------
        model_name: str
            The name of the model you want to add the params to.
        parameter_name: str | None
            The name of the param.
        parameter_vals: list(int) | None
            The list of the values.

        Returns
        -------
        None
        """
        self.selector.param_mapping[model_name][parameter_name] = parameter_vals

    def change_parameters_in_selector_model(self, model_name, parameter_name: str, parameter_vals: list[int]) -> None:
        """
        Replaces params in the selector param mapping with new ones.

        Parameters
        ----------
        model_name: str
            The name of the model you want to add the params to.
        parameter_name: str | None
            The name of the param.
        parameter_vals: list(int) | None
            The list of the values.

        Returns
        -------
        None
        """
        self.selector.param_mapping[model_name] = {parameter_name: parameter_vals}

    def find_best_models(self, n_best: int = 3) -> list:
        """
        Creates a dict of best performing models on the dataframe.

        Parameters
        ----------
        n_best: int
            Number of best performing models you want to get

        Returns
        -------
        None
        """
        self.best_models = self.selector.find_best_performing_models(n_best)
        return self.best_models

    #Todo: change name
    def metric_test_and_fitting_best_models(self, test_metric: str, n_best: int = 2) -> None:
        """
        Gets best performing models on the selected metric.

        Parameters
        ----------
        test_metric: str
            Best soothing metric for your classification problem
        n_best: int
            Number of best performing models you want to get

        Returns
        -------
        None
        """
        metric_function_mapping = {
            "accuracy": accuracy_score,
            "recall": recall_score,
            "f1": f1_score,
            "precision": precision_score,
        }

        if test_metric not in metric_function_mapping:
            raise ValueError("Invalid test metric. Choose from: 'accuracy', 'recall', 'f1', 'precision'.")

        metric_function = metric_function_mapping[test_metric]

        for model_dict in self.best_models:
            params = model_dict["params"]
            model = model_dict["model_class"](**params)

            model.fit(self.selector.X_train, self.selector.y_train)

            pipeline = Pipeline([("transformer", self.transformer), ("model", model)])
            y_pred = pipeline.predict(self.selector.X_test)

            if metric_function == "accuracy":
                score = metric_function(self.selector.y_test, y_pred)
            else:
                score = metric_function(self.selector.y_test, y_pred, average="micro")

            self.model_info.append({"pipeline": pipeline, "score": score})
        self.model_info = utils_find_best_performing_models(self.model_info, n_best)

    def print_model_info(self) -> None:
        """
        print the model_info dict with all params and scores.

        Returns
        -------
        None
        """
        for x in self.model_info:
            print(f"{x['name']} params are {x['model']}")
            print(f"{x['name']} got a score of {x['score']}")

    def create_model_pipeline(self, index: int) -> Pipeline:
        """
        Creates a pipeline for the selected model (either name of index is required).

        Parameters
        ----------
        index: int | None
            The index of the model in the model info list.

        Returns
        -------
        Pipeline
            model's pipeline.
        """
        # if not isinstance(index, int) or index > len(self.model_info) - 1:
        #     raise ValueError("Wrong index given")
        #
        # self.encoder = self.preprocessor.encoder
        #
        # model = self.model_info[index]["model"]
        # pipeline = Pipeline([("transformer", self.transformer), ("model", model)])
        #
        # return pipeline
        return self.model_info[index]["pipeline"]

    def save_model_to_pickle_file(
        self, file_path: str | Path, index: int = None, model_pipeline: Pipeline = None
    ) -> None:
        """
        Saves a model or pipeline to a pickle file (either index or model_pipeline required).

        Parameters
        ----------
        file_path: str | Path
            The path in which you want to save the file.

        index: int
            The index of the model in the model info list.

        model_pipeline: Pipeline
            If you created a pipeline it can be passed here to save.

        Returns
        -------
        None
        """
        if index:
            pipeline = self.create_model_pipeline(index=index)
        elif model_pipeline:
            pipeline = model_pipeline
        else:
            raise ValueError("you need to give either index or pipeline")

        with open(file_path, "wb") as file:
            pickle.dump(pipeline, file)

        print(f"Pipeline saved to {file_path}")

    @staticmethod
    def from_pickle_to_pipeline(file_path: str | Path) -> Pipeline:
        """
        Reads pickle files and turns them back int pipelines

        Parameters
        ----------
        file_path: str | Path
            The path to the file.

        Returns
        -------
        Pipeline: pipeline read from bytes
        """
        with open(file_path, "rb") as file:
            pipeline = pickle.load(file)

        return pipeline
