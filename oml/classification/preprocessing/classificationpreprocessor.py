import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from oml import logger
from oml.machine_learning_components.base_preprocessor import BasePreprocessor



class ClassificationPreprocessor(BasePreprocessor):
    """
    Class used for preprocessing dataframes.

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
        self.target_mapping = {}

    def transform(self) -> ColumnTransformer:
        """
        Imputes and encodes data according to the strategy.

        Returns
        -------
        ColumnTransformer: Transformer for the data.
        """
        cat_features = list(self.data.select_dtypes(include=["object", "category"]).columns)
        num_features = list(self.data.select_dtypes(include=["number"]).columns)

        cat_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

        num_transformer = Pipeline([("imputer", SimpleImputer(strategy=self.strategy))])

        transformer = ColumnTransformer(
            [
                ("num", num_transformer, num_features),
                ("cat", cat_transformer, cat_features),
            ],
        )

        self.X_train = transformer.fit_transform(self.X_train)

        logger.info("X_train successfully transformed.")

        return transformer
