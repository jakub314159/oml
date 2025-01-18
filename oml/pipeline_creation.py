import pandas as pd

from oml.preprocessing.preprocessor import Preprocessor, SimpleImputer
from oml.model_selection.model_selector import ModelSelector


class PipelineCreator:
    def __init__(self, dataframe: pd.DataFrame, target_column_name: str, strategy: str = "median"):
        self.preprocessor = Preprocessor(dataframe, target_column_name, strategy)
        self.imputer: SimpleImputer = None
        self.selector: ModelSelector = None

        self.best_models = {}

    def dataframe_preprocessing(self, test_df_size: float) -> None:
        """
        Does preprocessing on the dataframe.

        Parameters
        ----------
        test_df_size: float
            The size of the test df.
        Returns
        -------
        None
        """
        self.preprocessor.remove_invalid_data()
        self.preprocessor.data_target_split()
        self.preprocessor.get_dummies()
        self.preprocessor.map_target()
        self.preprocessor.train_test_split(test_size=test_df_size)
        self.preprocessor.impute()

        self.imputer = self.preprocessor.imputer

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
        self.selector.model_list = [x for x in self.selector.models_mapping.keys()]

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

    def find_best_performing_models(self, n_best: int = 3) -> None:
        """
        Creates a dict of best performing models on the dataframe.

        Parameters
        ----------
        n_best: int
            The amount of best models you want to get

        Returns
        -------
        None
        """
        self.best_models = self.selector.find_best_performing_models(n_best)
