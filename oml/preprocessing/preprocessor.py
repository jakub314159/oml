import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
        self._raw_df = dataframe
        self.df = dataframe

        self.target_column_name = target_column_name
        self.strategy = strategy

        self.data = None
        self.target = None
        self.target_mapping = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def __repr__(self):
        return f"{self.__class__.__name__}(dataframe={self.df}, target_column_name={self.target_column_name}, strategy={self.strategy})"

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

        return transformer
