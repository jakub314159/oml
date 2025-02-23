# Welcome to OML

Welcome to **OML** (Optimized Machine Learning), a project designed to improve your machine learning workflows. Whether you're a beginner or an experienced data scientist, OML provides tools and utilities to simplify data preprocessing, model selection, and pipeline creation.

---

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
|    |    | -- utils_finding_best_models.py  
|    | -- __init__.py  
|    | -- pipeline_creation.py  
| -- tests  
|    | -- functional
|    |    | -- test_model_selection  
|    |    |    | -- __init__.py  
|    |    |    | -- test_breast_cancer.py  
|    |    |    | -- test_iris_df.py
|    |    | -- test_pipeline_creation  
|    |    |    | -- __init__.py  
|    |    |    | -- test_breast_cancer.py  
|    |    |    | -- test_iris_df.py  
|    | -- unit  
|    |    | -- __init__.py  
|    |    | -- test_preprocessor.py  
|    | -- __init__.py  
| -- mkdocs.yml
| -- pyproject.toml
| -- README.md
```
###
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
>Dataframe before preprocessing
```
     x    y
0    0    a
1    1    a
2    2  NaN
3    3    b
4    4    b
5    5    b
6    6  NaN
7    7    c
8    8    c
9    9  NaN
10  10    a
11  11    a
12  12    a
13  13    b
14  14    b
15  15    b
16  16    c
17  17    c
18  18    c
19  19    c
20  20    a
21  21    a
22  22    a
23  23    b
24  24    b
25  25    b
26  26    c
27  27    c
28  28    c
29  29    c
```

```python
import pandas as pd
import numpy as np

from oml.classification.preprocessing import Preprocessor

df = pd.DataFrame({"x": [x for x in range(30)],
                   "y": ["a", "a", np.nan, "b", "b", "b", np.nan, "c", "c", np.nan] + [x for x in
                                                                                       "aaabbbccccaaabbbcccc"]})

preprocessor = Preprocessor(dataframe=df, target_column_name="y")

preprocessor.remove_invalid_data()
preprocessor.data_target_split()
preprocessor.train_test_split()
```
>Data and target from df
```
Data:      Target:
     x          y
0    0      0   0
1    1      1   0
2    3      2   1
3    4      3   1
4    5      4   1
5    7      5   2
6    8      6   2
7   10      7   0
8   11      8   0
9   12      9   0
10  13      10  1
11  14      11  1
12  15      12  1
13  16      13  2
14  17      14  2
15  18      15  2
16  19      16  2
17  20      17  0
18  21      18  0
19  22      19  0
20  23      20  1
21  24      21  1
22  25      22  1
23  26      23  2
24  27      24  2
25  28      25  2
26  29      26  2
```
## Example use of ModelSelector
>Example Input

```python
from sklearn.datasets import load_iris
import pandas as pd

from oml.classification.model_selection.model_selector import ModelSelector
from oml.classification.preprocessing import Preprocessor

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

preprocessor = Preprocessor(dataframe=iris_df, target_column_name="species")
preprocessor.remove_invalid_data()
preprocessor.data_target_split()
preprocessor.train_test_split()

selector = ModelSelector(preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test)
best_models = selector.find_best_performing_models(n_best=3)
print(best_models)
```
>Example Output
```
[
    {
        'model_name': 'KNN', 
        'model_class': <class 'sklearn.neighbors._classification.KNeighborsClassifier'>, 
        'params': {'n_neighbors': 5}, 
        'score': np.float64(0.9583333333333334)}, 
    {
        'model_name': 'logistic_regression', 
        'model_class': <class 'sklearn.linear_model._logistic.LogisticRegression'>, 
        'params': {'max_iter': 50}, 
        'score': np.float64(0.95)}, 
    {
        'model_name': 'logistic_regression', 
        'model_class': <class 'sklearn.linear_model._logistic.LogisticRegression'>, 
        'params': {'max_iter': 80}, 
        'score': np.float64(0.95)
    }
]
```
## Example use of PipelineCreator
>Example Input

```python
import pandas as pd
from sklearn.datasets import load_iris

from oml.classification.pipeline_creation import PipelineCreator

iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["species"] = iris.target
iris_df["species"] = iris_df["species"].map(dict(enumerate(iris.target_names)))

pipeline_creator = PipelineCreator(dataframe=iris_df, target_column_name="species")

pipeline_creator.dataframe_preprocessing(train_test_split_size=0.2)
pipeline_creator.prepare_model_selector()

pipeline_creator.find_best_models(n_best=3)
pipeline_creator.fit_best_pipelines(test_metric="recall")

pipeline_creator.print_pipeline_info()

pipeline = pipeline_creator.get_pipeline(index=0)
print(pipeline)
```
>Example Output
```
logistic_regression params are LogisticRegression(max_iter=50)
logistic_regression got a score of 0.9666666666666667
logistic_regression params are LogisticRegression(max_iter=80)
logistic_regression got a score of 0.9666666666666667

Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                ('model', LogisticRegression(max_iter=50))])
```