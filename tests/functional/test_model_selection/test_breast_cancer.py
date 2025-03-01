from sklearn.datasets import load_breast_cancer

from oml.classification.model_selection.model_selector import ClassificationModelSelector
from oml.classification.preprocessing.preprocessor import ClassificationPreprocessor


def test_model_selector_on_breast_cancer():
    all_data = load_breast_cancer(as_frame=True)
    data = all_data["data"]
    target = all_data["target"]

    preprocessor = ClassificationPreprocessor(all_data, None)
    preprocessor.data = data
    preprocessor.target = target
    preprocessor.train_test_split()
    preprocessor.transform()

    selector = ClassificationModelSelector(preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test)
    best_models = selector.find_best_performing_models(n_best=3)

    assert len(best_models) == 3

    for model in best_models:
        assert 1 >= model["score"] >= 0
