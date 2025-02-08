from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from oml.output_prediction import get_predict_function


@pytest.fixture
def mocked_pred_func(mocker) -> Callable:
    pipeline_creator = mocker.patch("oml.output_prediction.PipelineCreator")
    fake_pipeline = mocker.MagicMock(Pipeline)
    pipeline_creator.from_pickle_to_pipeline.return_value = fake_pipeline
    fake_pipeline.predict.return_value = ["Ghost"]
    func = get_predict_function(Path())

    return func


class TestOutputPrediction:
    def test_get_predict_function(self, mocked_pred_func):
        prediction = mocked_pred_func(["xd", "xd"])
        assert prediction == ["Ghost", "Ghost"]

    def test_get_predict_function_with_pd_df(self, mocked_pred_func):
        prediction = mocked_pred_func(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))
        assert prediction == ["Ghost", "Ghost"]


# pytest tests/unit --cov=oml --cov-report=term-missing
