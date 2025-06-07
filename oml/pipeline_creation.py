from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, root_mean_squared_error

from oml.machine_learning_components.base_preprocessor import BasePreprocessor
from oml.classification.model_selection.model_selector import ClassificationModelSelector
from oml.regression.model_selection.model_selector import RegressionModelSelector
from oml.utils.utils_finding_best_models import utils_find_best_performing_models
from oml import logger


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

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column_name: str,
        learning_type: str,
        score_metric: str,
        strategy: str = "median",
    ):
        self.preprocessor = BasePreprocessor(dataframe, target_column_name, strategy)
        self.selector: ClassificationModelSelector | RegressionModelSelector = None
        self.learning_type = learning_type
        self.score_metric = score_metric

        if self.learning_type not in ["regression", "classification"]:
            raise ValueError("goal is incorrect, You have to choose between 'regression' and 'classification'")

        self.transformer: ColumnTransformer | None = None

        self.best_models = {}
        self.pipeline_info = []

        self.metric_function_mapping = {
            "accuracy": accuracy_score,
            "recall": recall_score,
            "f1": f1_score,
            "precision": precision_score,
            "rmse": root_mean_squared_error,
        }

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
        logger.info("Preprocessing starting.")

        self.preprocessor.remove_invalid_data()
        self.preprocessor.data_target_split()

        self.preprocessor.train_test_split(test_size=train_test_split_size)

        self.transformer = self.preprocessor.transform()

        logger.info("Preprocessing done.")

    def prepare_model_selector(self) -> None:
        """
        Prepares model selector for classification.

        Returns
        -------
        None
        """
        match self.learning_type:
            case "classification":
                self.selector = ClassificationModelSelector(
                    self.preprocessor.X_train,
                    self.preprocessor.X_test,
                    self.preprocessor.y_train,
                    self.preprocessor.y_test,
                    self.metric_function_mapping[self.score_metric],
                )
            case "regression":
                self.selector = RegressionModelSelector(
                    self.preprocessor.X_train,
                    self.preprocessor.X_test,
                    self.preprocessor.y_train,
                    self.preprocessor.y_test,
                    self.metric_function_mapping[self.score_metric],
                )
            case _:
                raise ValueError("Wrong goal given.")

        logger.info("ModelSelector is prepared.")

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
        logger.info("model added.")

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
        logger.info("parameters added.")

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
        logger.info("parameters changed.")

    def find_best_models(self, n_best: int = 3) -> list:
        """
        Creates a dict of best performing models on the dataframe.

        Parameters
        ----------
        n_best: int
            Number of best performing models you want to get

        Returns
        -------
        list: Best models list
        """
        self.best_models = self.selector.find_best_performing_models(n_best)

        return self.best_models

    def fit_best_pipelines(self, n_best: int = 2) -> None:
        """
        Gets best performing models on the selected metric.

        Parameters
        ----------
        n_best: int
            Number of best performing models you want to get

        Returns
        -------
        None
        """
        logger.info("Starting pipeline fitting.")

        metric_function = self.metric_function_mapping[self.score_metric]

        for model_dict in self.best_models:
            params = model_dict["params"]
            model = model_dict["model_class"](**params)

            model.fit(self.selector.X_train, self.selector.y_train)

            pipeline = Pipeline([("transformer", self.transformer), ("model", model)])
            y_pred = pipeline.predict(self.selector.X_test)

            match self.learning_type:
                case "classification":
                    if self.score_metric == "accuracy":
                        score = metric_function(self.selector.y_test, y_pred)
                    else:
                        score = metric_function(self.selector.y_test, y_pred, average="micro")
                case "regression":
                    score = metric_function(self.selector.y_test, y_pred)
                case _:
                    raise ValueError("incorrect goal.")

            # pylint: disable=possibly-used-before-assignment
            self.pipeline_info.append({"pipeline": pipeline, "score": score})

        match self.learning_type:
            case "classification":
                self.pipeline_info = utils_find_best_performing_models(self.pipeline_info, n_best)
            case "regression":
                self.pipeline_info = utils_find_best_performing_models(self.pipeline_info, n_best, True)

        logger.info("pipelines fitted.")

    def get_pipeline(self, index: int) -> Pipeline:
        """
        gets a pipeline form the pipeline info.

        Parameters
        ----------
        index: int
            The index of the model in the model info list.

        Returns
        -------
        Pipeline
            model's pipeline.
        """
        return self.pipeline_info[index]["pipeline"]

    def save_pipeline_to_pickle_file(
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
        if index is not None:
            pipeline = self.get_pipeline(index=index)
        elif model_pipeline:
            pipeline = model_pipeline
        else:
            raise ValueError("you need to give either index or pipeline")

        with open(file_path, "wb") as file:
            pickle.dump(pipeline, file)

        logger.info("Pipeline saved to %s", file_path)

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

    @staticmethod
    def from_pickle_to_bytes(file_path: str | Path) -> Pipeline:
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
            pipeline_bytes = file.read()

        return pipeline_bytes
