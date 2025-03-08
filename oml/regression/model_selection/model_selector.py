from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from oml.machine_learning_components.base_model_selector import BaseModelSelector


class RegressionModelSelector(BaseModelSelector):
    """
    Class used for selecting the best performing models.

    Parameters
    ----------
    X_train: np.array
        X_train from train_test_split

    X_test: np.array
        X_test from train_test_split

    y_train: np.array
        y_train from train_test_split

    y_test: np.array
        y_test from train_test_split
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        self.models_mapping = {
            "linear_regression": LinearRegression,
            "decision_tree_regression": DecisionTreeRegressor,
        }

        self.model_list = list(self.models_mapping.keys())

        self.param_mapping = {
            "linear_regression": {"fit_intercept": [True, False]},
            "decision_tree_regression": {"max_depth": [2, 3, 5, 7, 8, 10, 12]},
        }
