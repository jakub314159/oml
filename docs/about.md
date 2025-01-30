# OML - Optimized Machine Learning

## What does it do?
This project's goal is to develop a framework that simplifies the process of building, testing, and deploying machine learning pipelines for classification problems.

## Why was it created?
This project was build to create pipelines containing the most optimal model for the classification problem you are facing.

## What can it do?
This project can:  
- turn a `pandas.Dataframe` into data and target dataframes.  
- remove invalid information from data and target.  
- map the target according to the user's preference.  
- make a train-test split on the data.  
- test out possible models for classification with different parameters and select the best ones.  
- make pipelines out of the best model configurations.  
- save the pipeline into a file in pickle format.  
- read pickle files and turn them into pipelines.  

## What does it implement?
The project implement 3 unique classes:
- `Preprocessor`: the class that prepares data for model selection.
- `ModelSelector`: the class that selects the optimal model according to the users preference.
- `PipelineCreator`: the class that manages the 2 above and is used for creating pipelines and saving them into files.

## dependencies:
- scikit-learn
- pandas
- numpy

## The Author
Jakub Pietrzak