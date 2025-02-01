import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class Preprocessor:
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
        self.df = dataframe
        self._raw_df = dataframe
        self.target_column_name = target_column_name
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self.data = None
        self.target = None
        self.target_mapping = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def __repr__(self):
        return f"{self.__class__.__name__}(dataframe={self.df}, target_column_name={self.target_column_name}, strategy={self.imputer.strategy})"

    def remove_invalid_data(self) -> None:
        """
        Removes all rows that don't have target.

        Returns
        -------
        None
        """
        for index, value in self.df[self.target_column_name].items():
            if pd.isna(value):
                self.df = self.df.drop(index)

        self.df = self.df.reset_index(drop=True)

        if len(self.df) <= 20:
            raise ValueError("Dataframe needs more than 20 valid rows.")

    def data_target_split(self) -> None:
        """
        Splits dataframe into 2 dataframes: data, target.

        Returns
        -------
        None
        """
        self.target = self.df[[self.target_column_name]]
        self.data = self.df.drop(columns=[self.target_column_name])

    # def get_dummies(self) -> None:
    #     """
    #     Pandas get_dummies on data.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     changeable = self.data.select_dtypes(include=["object", "category"])
    #
    #     if len(changeable.columns) > 0:
    #         encoded_array = self.encoder.fit_transform(changeable)
    #         encoded_df = pd.DataFrame(encoded_array, columns=self.encoder.get_feature_names_out())
    #         self.data = self.data.drop(columns=changeable.columns).reset_index(drop=True)
    #         self.data = pd.concat([self.data, encoded_df], axis=1)

    def map_target(self) -> None:
        """
        Maps target and prepares it for classification.

        Returns
        -------
        None
        """
        if not all([isinstance(x, int) for x in self.target]):
            if any([isinstance(x, int) for x in self.target]) and any([isinstance(x, str) for x in self.target]):
                raise ValueError("The target column is not properly represented.")

            mapping_val = 0
            for value in self.target[self.target_column_name]:
                if value not in self.target_mapping:
                    self.target_mapping[value] = mapping_val
                    mapping_val += 1

            self.target[self.target_column_name] = self.target[self.target_column_name].map(self.target_mapping)

    def train_test_split(self, test_size: float | None = 0.2) -> None:
        """
        Splits data and target into train and test according to the test_size.

        Parameters
        ----------
        test_size : float | None
            size of the tests (basic is 0.2)

        Returns
        -------
        None
        """
        a, b, c, d = train_test_split(self.data, self.target, test_size=test_size, stratify=self.target)

        self.X_train = a
        self.X_test = b
        self.y_train = c
        self.y_test = d

    # def impute(self) -> None:
    #     """
    #     If a value in data is None or np.nan it sets it according to the SimpleImputer strategy.
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     self.imputer.fit(self.X_train)
    #     self.X_train = self.imputer.transform(self.X_train)

    def transform(self) -> ColumnTransformer:
        cat_features = [x for x in self.data.select_dtypes(include=["object", "category"]).columns]
        num_features = [x for x in self.data.select_dtypes(include=['number']).columns]

        cat_transformer = Pipeline(
            [("encoder", OneHotEncoder(handle_unknown="ignore"))]  # Handle unseen categories in test data
        )

        num_transformer = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

        transformer = ColumnTransformer(
            [
                ("num", num_transformer, num_features),
                ("cat", cat_transformer, cat_features),
            ],
        )

        self.X_train = transformer.fit_transform(self.X_train)

        return transformer
