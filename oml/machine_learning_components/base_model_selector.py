from itertools import product

from sklearn.model_selection import cross_val_score

from oml.utils.utils_finding_best_models import utils_find_best_performing_models


class BaseModelSelector:
    """
    The parent of ModelSelector used for creating classification and regression versions.

    Attributes
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

    def __init__(self, X_train, X_test, y_train, y_test, score_metric):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train.to_numpy().flatten()
        self.y_test = y_test.to_numpy().flatten()

        self.score_metric = score_metric

        self.model_scores = []

        self.n_splits = 5

        self.models_mapping = {}
        self.model_list = []

        self.param_mapping = {}

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

            scores = cross_val_score(model, self.X_train, self.y_train, cv=self.n_splits, scoring=self.score_metric)
            score = sum(scores) / len(scores)

            self.model_scores.append(
                {
                    "model_name": model_name,
                    "model_class": self.models_mapping[model_name],
                    "params": combination,
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
