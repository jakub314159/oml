from sklearn.metrics import recall_score
from sklearn.metrics._scorer import _Scorer

from oml.machine_learning_components.base_model_selector import BaseModelSelector


def test_get_scorer(mock_train_test_data):
    X_train, X_test, y_train, y_test = mock_train_test_data
    base_model_selector = BaseModelSelector(X_train, X_test, y_train, y_test, recall_score)

    scorer = base_model_selector._get_scorer()

    assert isinstance(scorer, _Scorer)
