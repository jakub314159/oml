from pathlib import Path

import pandas as pd

from oml.regression.model_selection.model_selector import RegressionModelSelector
from oml.machine_learning_components.base_preprocessor import BasePreprocessor

_here = Path(__file__).parent


def test_regression_model_selector():
    data = pd.read_csv(_here.parent / "test_data" / "weatherHistory.csv")
    data = data.drop(columns=["Formatted Date"])
    data = data[:300]

    preprocessor = BasePreprocessor(dataframe=data, target_column_name="Temperature (C)")

    preprocessor.remove_invalid_data()
    preprocessor.data_target_split()
    preprocessor.train_test_split()
    preprocessor.transform()

    selector = RegressionModelSelector(
        preprocessor.X_train,
        preprocessor.X_test,
        preprocessor.y_train,
        preprocessor.y_test,
        "neg_root_mean_squared_error",
    )

    best_models = selector.find_best_performing_models(n_best=2)

    assert len(best_models) == 2

    for model in best_models:
        assert model["score"] <= 0
