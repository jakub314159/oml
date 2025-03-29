from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from oml.machine_learning_components.base_model_selector import BaseModelSelector


class ClassificationModelSelector(BaseModelSelector):
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

    def __init__(self, X_train, X_test, y_train, y_test, score_function):
        super().__init__(X_train, X_test, y_train, y_test, score_function)
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

        self.n_splits = min(Counter(self.y_train).values()) if min(Counter(self.y_train).values()) < 5 else 5
