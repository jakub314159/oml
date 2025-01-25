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
## Importing `PipelineCreator` and necessary modules
```python
import pandas as pd
from pathlib import Path
from oml.pipeline_creation import PipelineCreator
```

####
## Prepare your data and Set up `Pipelinecreator`
```python
pipeline_creator = PipelineCreator(
    dataframe= "the dataframe that will be split into data and target", 
    target_column_name= "name of the column you want to be the target", 
    strategy= "the strategy for the imputer for example median"
)
```
####
## Preprocess the data and prepare `ModelSelector`
```python
pipeline_creator.dataframe_preprocessing(
    test_df_size= "from 0 to 1 the size of the test"
)
pipeline_creator.prepare_model_selector()
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

####
## Finding best models
```python
pipeline_creator.find_best_performing_models(
    n_best = "number of best performing models you want to get"
)
```
####
## Evaluating chosen models with the selected metric
```python
pipeline_creator.get_n_best_scoring_models(
    test_metric= "best soothing metric for your classification problem", 
    n_best= "number of best performing models you want to get"
)
```

####
## Create a pipeline for the chosen model
```python
# If you are not sure about which model to chose you can see all models and scores
pipeline_creator.get_best_performing_models()

# Creating a pipeline (either name of index is required)
pipeline = pipeline_creator.create_model_pipeline(
    name= "(optional) The name of the model", 
    index= "(optional) The index of the model in the model info list"
)

# pipeline is now your completed pipeline with model and imputer
```

####
## Save the model into pickle format
```python
# either name or index or model_pipeline required

pipeline_creator.save_model_to_pickle_file(
    file_path= "path to the file you want to save the pipeline",
    name= "(optional) The name of the model", 
    index= "(optional) The index of the model in the model info list",
    model_pipeline= "If you created a pipeline it can be passed here to save"
)
```
