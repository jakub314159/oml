from itertools import product
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

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

        self.n_splits = min(Counter(self.y_train).values()) if min(Counter(self.y_train).values()) < 5 else 5

        self.models_mapping = {
            "logistic_regression": LogisticRegression,
            "KNN": KNeighborsClassifier,
            "decision_tree_classification": DecisionTreeClassifier,
            "random_forests": RandomForestClassifier,
            "svc": SVC,
        }

        self.model_list = list(self.models_mapping.keys())

        self.param_mapping = {
            "logistic_regression": {"max_iter": [50, 80, 100, 120, 150]},
            "KNN": {"n_neighbors": [4, 5, 8, 10, 12, 13, 16, 23, 25]},
            "decision_tree_classification": {"max_depth": [2, 3, 5, 7, 9]},
            "random_forests": {"n_estimators": [90, 100, 110, 120, 130]},
            "svc": {"degree": [2, 3, 4, 5, 6]},
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

            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.n_splits)
            score = sum(scores) / len(scores)

            self.model_scores.append(
                {
                    "model_name": model_name,
                    "model_class": self.models_mapping[model_name],
                    "params": combination,
                    "score": score,
                }
            )

    def find_best_performing_models(self, n_best: int = 3) -> list[dict]:
        """
        Sorts all the model scores and return n_best.

        Parameters
        ----------
        n_best: int = 3
            The number of scores returned.

        Returns
        -------
        list[dict]
            The list of the best models and their info.
        """
        for model_name in self.model_list:
            self.get_all_model_scores(model_name)

        return utils_find_best_performing_models(self.model_scores, n_best)
