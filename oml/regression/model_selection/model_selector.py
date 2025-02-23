from itertools import product
from collections import Counter

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from oml.utils.utils_finding_best_models import utils_find_best_performing_models


class ModelSelector:
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
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.to_numpy().flatten()
        self.y_test = y_test.to_numpy().flatten()

        self.model_scores = []

        self.models_mapping = {
            "linear_regression": LinearRegression,
            "polynomial_regression": PolynomialFeatures,
            "decision_tree_regression": DecisionTreeRegressor,
        }

        self.model_list = list(self.models_mapping.keys())

        self.param_mapping = {
            "linear_regression": {"fit_intercept": [True, False]},
            "polynomial_regression": {"degree": [2, 3, 5, 7, 8, 10]},
            "decision_tree_regression": {"max_depth": [2, 3, 5, 7, 8, 10, 12]},
        }
    def get_all_model_scores(self, model_name: str) -> None:
        """
        Gets scores from all parameters on the train test values.

        Parameters
        ----------
        model_name: str
            The name of the model that will be tested.

        Returns
        -------
        None
        """
        params = self.param_mapping[model_name]
        combinations = [dict(zip(params.keys(), combination)) for combination in product(*params.values())]
        class_name = self.models_mapping[model_name]

        for combination in combinations:
            model = class_name(**combination)

            model.fit(self.X_train, self.y_train)

            prediction = model.predict(self.X_test)
            score = r2_score(self.X_test, prediction)

            self.model_scores.append(
                {
                    "model_name": model_name,
                    "model": model,
                    "score": score,
                }
            )

    def find_best_performing_models(self, n_best: int = 3) -> list:
        """
        Sorts all the model scores and return n_best.

        Parameters
        ----------
        n_best: int = 3
            The number of scores returned.

        Returns
        -------
        dict: The scores
        """
        for model_name in self.model_list:
            self.get_all_model_scores(model_name)

        return utils_find_best_performing_models(self.model_scores, n_best)
