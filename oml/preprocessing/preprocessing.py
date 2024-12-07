from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

iris_df["species"] = iris.target
iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))


class Preprocessor:
    def __init__(self, dataframe: pd.DataFrame, target, strategy: str = "median"):
        self.df = dataframe
        self.target = target
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)

    def remove_invalid_data(self):
        for index, value in self.df[self.target].items():
            if pd.isna(value):
                self.df = self.df.drop(index)

    def data_target_split(self):
        df_dict = self.df.to_dict()

        self.target = df_dict.pop(self.target)
        self.data = pd.DataFrame.from_dict(df_dict)

    def impute(self):
        self.imputer.fit(self.data)
        self.data = self.imputer.transform(self.data)

    def get_dummies(self):
        self.data = pd.get_dummies(self.data)

    def target_mapping(self):
        names = {}

        a = 0
        for value in self.target.values():
            if value not in names:
                names[(str(value))] = a
                a += 1

        for key, val in self.target.items():
            self.target[key] = names[val]

        self.target = pd.DataFrame.from_dict(self.target, orient="index")
        return names

    def train_test_split(self, test_size: float | None = 0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=test_size)

        print(type(X_train))


if __name__ == "__main__":
    processor = Preprocessor(iris_df, "species")
    processor.remove_invalid_data()
    processor.data_target_split()
    processor.target_mapping()
    print(processor.train_test_split())
