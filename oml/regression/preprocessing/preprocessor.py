import pandas as pd

from oml.machine_learning_components.base_preprocessor import BasePreprocessor


class RegressionPreprocessor(BasePreprocessor):
    """
    Class used for preprocessing dataframes for regression problems.

    Parameters
    ----------
    dataframe : pd.Dataframe
        Pandas dataframe used for preprocessing.

    target_column_name: str
        Name of the pandas column you want to set as target.

    strategy: str
        Strategy for the SimpleImputer.
    """

    def __init__(self, dataframe: pd.DataFrame, target_column_name: str, strategy: str = "median"):
        super().__init__(dataframe, target_column_name, strategy)
