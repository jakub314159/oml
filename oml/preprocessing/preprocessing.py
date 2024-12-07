from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, dataframe: pd.DataFrame, target_column_name: str, strategy: str = "median"):
        """

        Parameters
        ----------
        dataframe : pd.Dataframe
            Pandas dataframe used for preprocessing.

        target_column_name: str
            Name of the pandas column you want to set as target.

        strategy: str
            Strategy for the SimpleImputer.
        """
        self.df = dataframe
        self._raw_df = dataframe
        self.target_column_name = target_column_name
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

        self.data = None
        self.target = None
        self.target_mapping = {}

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def __repr__(self):
        return f"{self.__class__.__name__}(dataframe={self.df}, target_column_name={self.target_column_name}, strategy={self.imputer.strategy})"

    def remove_invalid_data(self):
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

    def data_target_split(self):
        """
        Splits dataframe into data and target.

        Returns
        -------
        None
        """
        df_dict = self.df.to_dict()

        self.target = df_dict.pop(self.target_column_name)
        self.data = pd.DataFrame.from_dict(df_dict)

    def impute(self):
        """
        If a value in data is None or np.nan it sets it according to the SimpleImputer strategy.

        Returns
        -------
        None
        """
        self.imputer.fit(self.X_train)
        self.X_train = self.imputer.transform(self.X_train)

    def get_dummies(self):
        """
        Pandas get_dummies on data.

        Returns
        -------
        None
        """
        self.data = pd.get_dummies(self.data)

    def map_target(self):
        """
        Maps target and prepares it for classification.
        Makes target a dataframe.

        Returns
        -------
        None
        """
        a = 0
        for value in self.target.values():
            if value not in self.target_mapping:
                self.target_mapping[(str(value))] = a
                a += 1

        for key, val in self.target.items():
            self.target[key] = self.target_mapping[val]

        self.target = pd.DataFrame.from_dict(self.target, orient="index")

    def train_test_split(self, test_size: float | None = 0.2):
        """
        Splits data and target into train and test according to the test_size

        Parameters
        ----------
        test_size : float | None
            size of the tests (basic is 0.2)

        Returns
        -------
        None
        """
        a, b, c, d = train_test_split(self.data, self.target, test_size=test_size)

        self.X_train = a
        self.X_test = b
        self.y_train = c
        self.y_test = d

if __name__ == "__main__":
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

    iris_df["species"] = iris.target
    iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

    processor = Preprocessor(iris_df, "species")
    processor.remove_invalid_data()
    processor.data_target_split()
    processor.map_target()
    print(processor.train_test_split())