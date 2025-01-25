# Welcome to OML

Welcome to **OML** (Optimized Machine Learning), a project designed to improve your machine learning workflows. Whether you're a beginner or an experienced data scientist, OML provides tools and utilities to simplify data preprocessing, model selection, and pipeline creation.

---

## Commands

Here are the commands to get started with OML:


## Project layout
```bash
.  
| -- oml  
|    | -- model_selection  
|    |    | -- __init__.py  
|    |    | -- model_selector.py  
|    | -- preprocessing  
|    |    | -- __init__.py  
|    |    | -- preprocessor.py  
|    | -- utils  
|    |    | -- __init__.py  
|    |    | -- finding_best_models.py  
|    | -- __init__.py  
|    | -- pipeline_creation.py  
| -- tests  
|    | -- functional  
|    |    | -- __init__.py  
|    |    | -- test_breast_cancer.py  
|    |    | -- test_iris_df.py  
|    | -- unit  
|    |    | -- __init__.py  
|    |    | -- test_preprocessor.py  
|    | -- __init__.py  
```
## OML for better machine learning!
OML is the optimization for machine learning workflows. Here's how you can use it:  

- **Data Preprocessing**: Can by accessed by `PipelineCreator` class or created from `Preprocessor` class to preprocess your data effortlessly.   
    It handles invalid data removal, dummy encoding, target mapping, and splitting into training and testing sets.  
####
- **Model Selection**: Add and configure multiple models using `PipelineCreator` or `ModelSelector` class.  
    It allows you to test various hyperparameters and find the best-performing models for your classification project.  
####
- **Pipeline Creation**: Create complete machine learning pipelines with minimal effort by `PipelineCreator`.  
    Manage models and save them to .pkl files for reuse in production.  
####

## Example use of Preprocessor

## Example use of ModelSelector

## Example use of PipelineCreator
> Input
```python
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris


from oml.pipeline_creation import PipelineCreator

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))


pipeline_creator = PipelineCreator(dataframe=iris_df, target_column_name="species")

pipeline_creator.dataframe_preprocessing(test_df_size=0.2)
pipeline_creator.prepare_model_selector()


pipeline_creator.find_best_performing_models(n_best=3)

pipeline_creator.get_best_performing_models()

pipeline = pipeline_creator.create_model_pipeline(index=0)

file_path = Path(__file__)
pipeline_creator.save_model_to_pickle_file(file_path=file_path, model_pipeline= pipeline)
```
> Output
```python

```