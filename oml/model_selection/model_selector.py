from itertools import product
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from oml.utils.finding_best_models import find_best_performing_models


class ModelSelector:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.to_numpy().flatten()
        self.y_test = y_test.to_numpy().flatten()

        self.model_scores = []

        self.n_splits = min(min(Counter(self.y_train).values()), 5)

        self.models_mapping = {
            "logistic_regression": LogisticRegression,
            "KNN": KNeighborsClassifier,
            "decision_tree_classification": DecisionTreeClassifier,
            "random_forests": RandomForestClassifier,
            "svc": SVC,
        }

        self.model_list = [x for x in self.models_mapping.keys()]

        self.param_mapping = {
            "logistic_regression": {"max_iter": [50, 80, 100, 120, 150]},
            "KNN": {"n_neighbors": [4, 5, 8, 10, 12, 13, 16, 23, 25]},
            "decision_tree_classification": {"max_depth": [2, 3, 5, 7, 9]},
            "random_forests": {"n_estimators": [90, 100, 110, 120, 130]},
            "svc": {"degree": [2, 3, 4, 5, 6]},
        }

    def get_all_model_scores(self, model_name: str):
        params = self.param_mapping[model_name]
        combinations = [dict(zip(params.keys(), combination)) for combination in product(*params.values())]
        class_name = self.models_mapping[model_name]

        for combination in combinations:
            model = class_name(**combination)

            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.n_splits)
            score = sum(scores) / len(scores)

            self.model_scores.append({"model_name": model_name, "params": combination, "score": score})

    def find_best_performing_models(self, n_best: int = 3):
        for model_name in self.model_list:
            self.get_all_model_scores(model_name)

        return find_best_performing_models(self.model_scores, n_best)
