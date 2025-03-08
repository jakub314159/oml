# Tutorial: How to Use the `PipelineCreator` Class

This tutorial demonstrates how to use the `PipelineCreator` class for data preprocessing, model selection, pipeline creation, and saving pipelines to `.pkl` files.

---
## Setting up venv
####
```bash
python -m venv venv
source venv/Scripts/activate
```
>if "source venv/Scripts/activate" doesn't work try out another Pycharm environment

## Installing Required Packages

Make sure you have the following packages installed:

- **pandas**
- **scikit-learn**
- **numpy**


Install them using `pip`:


```bash
pip install oml[dev]
```

####
## Tutorial made with iris dataset from sklearn

```python
# Importing PipelineCreator and necessary modules:
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris

from oml.classification.pipeline_creation import PipelineCreator

# Prepare your data
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

# Set up PipelineCreator
pipeline_creator = PipelineCreator(
    dataframe=iris_df,
    target_column_name="species",
    strategy="median"
)

# Preprocess the data and prepare `ModelSelector`
pipeline_creator.dataframe_preprocessing(train_test_split_size=0.2)
pipeline_creator.prepare_model_selector()

# Finding best models
pipeline_creator.find_best_models(
    n_best=3
)

# Evaluating chosen models with the selected metric
pipeline_creator.fit_best_pipelines(
    test_metric="recall",
    n_best=2
)

# If you are not sure about which model to chose you can see all models and scores
pipeline_creator.print_pipeline_info()

# Creating a pipeline
pipeline = pipeline_creator.get_pipeline(
    index=0
)
# pipeline is now your completed pipeline with model and imputer


# Save the model into pickle format (either index or model_pipeline is required)
pipeline_creator.save_pipeline_to_pickle_file(
    file_path=Path(__file__),
    index=None,
    model_pipeline=pipeline
)
```
####
## Getting model pipeline from pickle
```python
pipeline = pipeline_creator.from_pickle_to_pipeline(
    file_path= Path(__file__)
)
```
####
## (Optional) You can add models or parameters to `ModelSelector`
```python
# Adding new models to selector
pipeline_creator.add_models_to_selector(
    model_name= "model name", 
    model_class = "model class for exaple 'LogisticRegression'", 
    parameter_name= "Optional: param name if you want to add one", 
    parameter_vals= "Optional: list of param values if you want to add one (requires param_name)"
)

# Adding parameters to model
pipeline_creator.add_parameters_to_selector_model(
    model_name= "existing model name", 
    parameter_name= "the name of the param added", 
    parameter_vals= "list of param values added"
)

# Changing parameters in a model
pipeline_creator.change_parameters_in_selector_model(
    model_name= "existing model name", 
    parameter_name= "the name of the param changed", 
    parameter_vals= "list of param values changed"
)
```
