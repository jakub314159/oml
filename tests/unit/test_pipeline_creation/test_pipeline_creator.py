from pathlib import Path
from unittest.mock import mock_open

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from oml.pipeline_creation import PipelineCreator
from oml.machine_learning_components.base_preprocessor import BasePreprocessor
from oml.classification.model_selection.model_selector import ClassificationModelSelector
from oml import logger


@pytest.fixture
def preprocessor(mocker, mock_train_test_data):
    preprocessor = mocker.MagicMock(BasePreprocessor)
    preprocessor.transform.return_value = mocker.MagicMock(ColumnTransformer)

    preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test = mock_train_test_data

    return preprocessor


@pytest.fixture
def selector(mocker, model_scores, mock_train_test_data):
    selector = mocker.MagicMock(ClassificationModelSelector)
    selector.find_best_performing_models.return_value = model_scores

    selector.models_mapping = {"b": "b"}
    selector.param_mapping = {"b": {"b": "b"}}

    selector.X_train, selector.X_test, selector.y_train, selector.y_test = mock_train_test_data

    return selector


@pytest.fixture
def model_scores(mocker):
    return [
        {
            "model_name": "KNN",
            "model_class": mocker.MagicMock(),
            "params": {"a": [1, 2, 3]},
            "score": 0.9,
        },
        {
            "model_name": "KNN",
            "model_class": mocker.MagicMock(),
            "params": {"a": [1, 2, 3]},
            "score": 0.85,
        },
    ]


@pytest.fixture
def pipeline(mocker):
    return mocker.MagicMock(Pipeline)


@pytest.fixture
def pipeline_creator(mock_dataframe, preprocessor, mocker, selector, model_scores, pipeline):
    mocker.patch("oml.machine_learning_components.base_preprocessor.BasePreprocessor", return_value=preprocessor)
    mocker.patch("oml.pipeline_creation.ClassificationModelSelector", return_value=selector)

    pipeline_creator = PipelineCreator(
        dataframe=mock_dataframe, target_column_name="target", learning_type="classification", score_metric="recall"
    )
    pipeline_creator.pipeline_info = [{"pipeline": pipeline}]

    return pipeline_creator


class TestPipelineCreator:
    def test_dataframe_preprocessing(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")
        pipeline_creator.dataframe_preprocessing()

        assert isinstance(pipeline_creator.transformer, ColumnTransformer)
        logger.info.assert_called_with("Preprocessing done.")

    def test_wrong_learning_type(self):
        with pytest.raises(ValueError) as e:
            PipelineCreator(1, 1, 1, 1)

    def test_prepare_model_selector_with_classification(self, pipeline_creator, mocker):
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

    def test_find_best_models(self, pipeline_creator, model_scores, mocker):
        mocker.patch.object(logger, "info")

        pipeline_creator.prepare_model_selector()
        best_models = pipeline_creator.find_best_models(2)

        assert best_models == model_scores

    @pytest.mark.parametrize("metric", ["accuracy", "recall"])
    def test_fit_best_pipelines(self, pipeline_creator, mocker, model_scores, metric):
        mocker.patch.object(logger, "info")

        accuracy_score_mock = mocker.MagicMock(return_value=1)
        recall_score_mock = mocker.MagicMock(return_value=1)

        pipeline_creator.metric_function_mapping = {"accuracy": accuracy_score_mock, "recall": recall_score_mock}

        pipeline_creator.pipeline_info = []

        pipeline_creator.prepare_model_selector()
        pipeline_creator.best_models = model_scores

        pipeline_creator.fit_best_pipelines()
        assert len(pipeline_creator.pipeline_info) > 0

    def test_get_pipeline(self, pipeline_creator):
        returned_pipeline = pipeline_creator.get_pipeline(0)

        assert pipeline_creator.pipeline_info[0]["pipeline"] == returned_pipeline

    @pytest.mark.parametrize("form", [{"index": 0}, {"model_pipeline": "pipeline"}])
    def test_save_pipeline_to_pickle_file(self, pipeline_creator, mocker, caplog, form):
        mocker.patch("oml.pipeline_creation.open")
        mocker.patch("oml.pipeline_creation.pickle.dump")

        path = "filepath"

        pipeline_creator.save_pipeline_to_pickle_file(file_path=path, **form)
        assert f"Pipeline saved to {path}" in caplog.text

    def test_save_pipeline_to_pickle_file_value_error(self, pipeline_creator, mocker):
        mocker.patch.object(logger, "info")

        with pytest.raises(expected_exception=ValueError) as e:
            pipeline_creator.save_pipeline_to_pickle_file(file_path=mocker.MagicMock(Path))

        assert str(e.value) == "you need to give either index or pipeline"

    def test_from_pickle_to_pipeline(self, pipeline_creator, mocker, pipeline):
        mocker.patch("oml.pipeline_creation.open")
        mocker.patch("oml.pipeline_creation.pickle.load", return_value=pipeline)

        path = mocker.MagicMock(Path)

        pipeline = pipeline_creator.from_pickle_to_pipeline(path)

        assert pipeline == pipeline_creator.pipeline_info[0]["pipeline"]

    def test_from_pickle_to_bytes(self, pipeline_creator, mocker):
        mock_file = mock_open(read_data="pipeline")
        mocker.patch("oml.pipeline_creation.open", mock_file)

        path = mocker.MagicMock(Path)
        bytes = pipeline_creator.from_pickle_to_bytes(path)

        assert bytes == "pipeline"
