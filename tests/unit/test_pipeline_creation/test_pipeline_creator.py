from pathlib import Path

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from oml.pipeline_creation import PipelineCreator
from oml.preprocessing.preprocessor import Preprocessor
from oml.model_selection.model_selector import ModelSelector
from oml import logger

@pytest.fixture
def preprocessor(mocker, mock_train_test_data):
    preprocessor = mocker.MagicMock(Preprocessor)
    preprocessor.transform.return_value = mocker.MagicMock(ColumnTransformer)

    preprocessor.X_train , preprocessor.X_test, preprocessor.y_train, preprocessor.y_test = mock_train_test_data

    return preprocessor

@pytest.fixture
def selector(mocker, model_scores):
    selector = mocker.MagicMock(ModelSelector)
    selector.find_best_performing_models.return_value = model_scores

    selector.models_mapping = {"b": "b"}
    selector.param_mapping = {"b": {"b": "b"}}

    return selector

@pytest.fixture
def model_scores():
    return [
        {
            "model_name": "KNN",
            "model_class": "KNN",
            "params": "knn_params",
            "score": 0.9,
        },
        {
            "model_name": "Logistic",
            "model_class": "Logistic",
            "params": "logistic_params",
            "score": 0.85,
        },
    ]


@pytest.fixture
def pipeline_creator(mock_dataframe, preprocessor, mocker, selector, model_scores):
    pipeline = mocker.MagicMock(Pipeline)

    mocker.patch("oml.pipeline_creation.Preprocessor", return_value= preprocessor)
    mocker.patch("oml.pipeline_creation.ModelSelector", return_value= selector)

    pipeline_creator = PipelineCreator(dataframe=mock_dataframe, target_column_name= "target")
    pipeline_creator.pipeline_info = [{"pipeline": pipeline}]

    return pipeline_creator

@pytest.fixture
def fit_pipeline_creator(pipeline_creator):
    pipeline_creator.pipeline_info = []

    return pipeline_creator


class TestPipelineCreator:
    def test_dataframe_preprocessing(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")
        pipeline_creator.dataframe_preprocessing()

        assert isinstance(pipeline_creator.transformer, ColumnTransformer)
        logger.info.assert_called_with("Preprocessing done.")

    def test_prepare_model_selector(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")

        pipeline_creator.prepare_model_selector()

        assert pipeline_creator.selector is not None
        logger.info.assert_called_with("ModelSelector is prepared.")

    def test_add_models_to_selector(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")

        pipeline_creator.prepare_model_selector()

        model_list_before = len(pipeline_creator.selector.models_mapping)

        pipeline_creator.add_models_to_selector("a", "a", "a", "a")

        assert len(pipeline_creator.selector.models_mapping) > model_list_before
        assert pipeline_creator.selector.models_mapping["a"] == "a"
        assert pipeline_creator.selector.param_mapping["a"] == {"a": "a"}

    def test_add_parameters_to_model_selector(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")

        pipeline_creator.prepare_model_selector()

        pipeline_creator.add_parameters_to_selector_model("b", "b", ["b", "b"])

        assert pipeline_creator.selector.param_mapping["b"] == {"b": ["b", "b"]}

    def test_change_parameters_in_selector_model(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")

        pipeline_creator.prepare_model_selector()

        pipeline_creator.change_parameters_in_selector_model("b", "b", "a")

        assert pipeline_creator.selector.param_mapping["b"] == {"b": "a"}

    def test_find_best_models(self, pipeline_creator, model_scores):
        best_models = pipeline_creator.find_best_models(2)

        assert best_models == model_scores

    def test_fit_best_pipelines(self, fit_pipeline_creator, mocker):
        mocker.patch.object(logger, "info")
        fit_pipeline_creator.fit_best_pipelines()
        assert len(fit_pipeline_creator.pipeline_info) > 0


    def test_get_pipeline(self, pipeline_creator):
        returned_pipeline = pipeline_creator.get_pipeline(0)

        assert pipeline_creator.pipeline_info[0]["pipeline"] == returned_pipeline

    def test_save_pipeline_to_pickle_file_index(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")

        path = mocker.MagicMock(Path)

        pipeline_creator.save_pipeline_to_pickle_file(file_path=path, index=0)
        logger.info.assert_called_with(f"Pipeline saved to {path}")

    def test_save_pipeline_to_pickle_file_pipeline(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")
    
        path = mocker.MagicMock(Path)
    
        pipeline_creator.save_pipeline_to_pickle_file(file_path=path, model_pipeline= mocker.MagicMock(Pipeline))
        logger.info.assert_called_with(f"Pipeline saved to {path}")

    def test_from_pickle_to_pipeline(self, pipeline_creator, mocker):
        path = mocker.MagicMock(Path)

        pipeline = pipeline_creator.from_pickle_to_pipeline(path)

        assert pipeline == pipeline_creator.pipeline_info[0]["pipeline"]
