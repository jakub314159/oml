import pandas as pd

from oml.machine_learning_components.base_preprocessor import BasePreprocessor


class RegressionPreprocessor(BasePreprocessor):
    def __init__(self, dataframe: pd.DataFrame, target_column_name: str, strategy: str = "median"):
        super().__init__(dataframe, target_column_name, strategy)
        self.df = self.df.select_dtypes(include=["number"])