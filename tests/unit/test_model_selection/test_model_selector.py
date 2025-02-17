import numpy as np
import pandas as pd
import pytest
from oml.model_selection.model_selector import ModelSelector
from oml.utils.utils_finding_best_models import utils_find_best_performing_models



@pytest.fixture
def model_selector(mock_train_test_data, mocker):
    X_train, X_test, y_train, y_test = mock_train_test_data

    mocker.patch("oml.model_selection.model_selector.cross_val_score", return_value=[0.8, 0.85, 0.9])

    return ModelSelector(X_train, X_test, y_train, y_test)


class TestModelSelector:
    def test_get_all_model_scores(self, model_selector, mocker):
        model_selector.get_all_model_scores("logistic_regression")

        assert len(model_selector.model_scores) > 0
        assert model_selector.model_scores[0]["model_name"] == "logistic_regression"
        assert model_selector.model_scores[0]["score"] == 0.85

    def test_find_best_performing_models(self, model_selector, mocker):
        mocker.patch(
            "oml.utils.utils_finding_best_models.utils_find_best_performing_models",
            return_value=["logistic_regression", "KNN", "random_forests"],
        )

        model_selector.model_scores = [
            {"model_name": "logistic_regression", "score": 0.75},
            {"model_name": "KNN", "score": 0.85},
            {"model_name": "random_forests", "score": 0.80},
        ]

        best_models = model_selector.find_best_performing_models(n_best=3)
        assert best_models == utils_find_best_performing_models(model_selector.model_scores, 3)
