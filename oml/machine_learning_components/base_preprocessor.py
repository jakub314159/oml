import pandas as pd
from sklearn.model_selection import train_test_split

from oml import logger


class BasePreprocessor:

    def __init__(self, dataframe: pd.DataFrame, target_column_name: str, strategy: str = "median"):
        self._raw_df = dataframe
        self.df = dataframe

        self.target_column_name = target_column_name
        self.strategy = strategy

        self.data = None
        self.target = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    def __repr__(self):
        return (
            f"{self.__class__.__name__}(dataframe={self.df}, target_column_name={self.target_column_name}, "
            f"strategy={self.strategy})"
        )

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

        logger.info("invalid data was removed.")

    def data_target_split(self) -> None:
        """
        Splits dataframe into 2 dataframes: data, target.

        Returns
        -------
        None
        """
        logger.info("Starting data_target_split")

        self.target = self.df[[self.target_column_name]]
        self.data = self.df.drop(columns=[self.target_column_name])

        logger.info("Splitting done.")


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
        logger.info(msg="starting train_test_split")

        a, b, c, d = train_test_split(self.data, self.target, test_size=test_size, stratify=self.target)

        self.X_train = a
        self.X_test = b
        self.y_train = c
        self.y_test = d

        logger.info(
            "train_test_split done. Training shape: %s, Testing shape: %s", self.X_train.shape, self.X_test.shape
        )
