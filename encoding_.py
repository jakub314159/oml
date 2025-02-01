import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Sample dataset
data = pd.DataFrame({
    'Age': [25, 30, 35, np.nan, 40],
    'Salary': [50000, 60000, 70000, 80000, np.nan],
    'City': ['New York', 'Paris', 'London', 'New York', 'Paris'],
    'Purchased': ['No', 'Yes', 'No', 'Yes', 'No']
})

# Split features and target
X = data.drop(columns=['Purchased'])
y = data['Purchased']

# Define feature types
categorical_features = ['City']
numerical_features = ['Age', 'Salary']

# Preprocessing for numerical data (handling missing values and scaling)
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data (OneHotEncoding)
cat_transformer = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Handle unseen categories in test data
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_features),
    ('cat', cat_transformer, categorical_features)
])

# Define full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

print("Pipeline successfully trained and used for predictions.")
