from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class ModelSelector:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model_scores = []

        self.model_list = ["logistic_regression", "KNN", "decision_tree_classification", "random_forests", "svc"]
        self._models_mapping = {
            "logistic_regression": LogisticRegression,
            "KNN": KNeighborsClassifier,
            "decision_tree_classification": DecisionTreeClassifier,
            "random_forests": RandomForestClassifier,
            "svc": SVC,
        }

        self.param_mapping = {
            "logistic_regression": {"max_iter": [50, 80, 100, 120, 150]},
            "KNN": {"n_neighbors": [4, 5, 8, 10, 12, 13, 16, 23, 25]},
            "decision_tree_classification": {"max_depth": [2, 3, 5, 7, 9]},
            "random_forests": {"n_estimators": [90, 100, 110, 120, 130]},
            "svc": {"degree": [2, 3, 4, 5, 6]},
        }

    def get_all_model_scores(self, model_name: str, scoring: str = "recall", cv: int = 5):
        params = self.param_mapping[model_name]
        combinations = [dict(zip(params.keys(), combination)) for combination in product(*params.values())]
        class_name = self._models_mapping[model_name]

        for combination in combinations:
            model = class_name(**combination)
            scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring=scoring)

            score = sum(scores) / len(scores)
            self.model_scores.append({"model_name": model_name, "params": combination, "score": score})


    def find_best_performing_models(self, scoring: str = "recall", n_best: int = 3):
        for model_name in self.model_list:
            self.get_all_model_scores(model_name, scoring)

        best_models = []
        
        x_best_scores = sorted([x["score"] for x in self.model_scores], reverse=True)[:n_best]
        print(x_best_scores)
        
        
        for param_dict in self.model_scores:
            score = param_dict["score"]
        
            if score in x_best_scores:
                best_models.append(param_dict)
        
        return best_models

if __name__ == "__main__":
    import pandas as pd

    from oml.preprocessing.preprocessor import Preprocessor

    df = pd.DataFrame({"x": [x for x in range(10)], "y": [x for x in "aaabbbcccc"]})

    preprocessor = Preprocessor(dataframe=df, target_column_name="y")
    preprocessor.data_target_split()
    preprocessor.train_test_split()

    selector = ModelSelector(preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test)
    selector.get_all_model_scores(model_name="KNN")

    print(selector.model_scores)


