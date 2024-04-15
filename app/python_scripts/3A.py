# %%
import pandas as pd
import dask.dataframe as dd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
import pickle

# %% [markdown]
# # Paths

# %%
# Data
processed_dir = os.path.join("..", "data", "processed")
final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')
minority_users_dt_pq = os.path.join(processed_dir, 'minority_users_dt.parquet')

# Model
models_dir = os.path.join("..", "models", "models")
model_dump_path = os.path.join(models_dir, 'dt_model.pkl')

# Figures
figures_dir = os.path.join("..", "materials", "reports", "figures", "3A")
dt_confusion_matrix_path = os.path.join(figures_dir, 'dt_confusion_matrix.png')
dt_feature_importance_path = os.path.join(
    figures_dir, 'dt_feature_importance.png')
dt_visualisation_path = os.path.join(figures_dir, 'dt_visualisation.png')
dt_train_confidence_score_path = os.path.join(
    figures_dir, 'dt_train_confidence_score.png')
dt_test_confidence_score_path = os.path.join(
    figures_dir, 'dt_test_confidence_score.png')

# %% [markdown]
# Import Dataset

# %%
# Read the multi-part Parquet dataset
fds = pd.read_parquet(final_dataset_path)

# %% [markdown]
# # Decision Tree / Random Forest

# %%
X = fds.drop(columns=['user', 'malicious'])
y = fds.malicious

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# %% [markdown]
# **Feature Normalisation**

# %%
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# **Hyperparameter Tuning**

# %%
dt = DecisionTreeClassifier(random_state=42)
# Define the hyperparameters to tune
param_grid = {
    'max_depth': [3, 5, 8, 10],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 3, 5],
    # 'criterion': ['gini', 'entropy', 'log_loss'], gini
    # 'splitter': ['best', 'random'], best
    'max_features': ['sqrt', 'log', None],
    # 'min_weight_fraction_leaf': [0.0, 0.1, 0.2], 0
    # 'max_leaf_nodes': [None, 2, 4],
    # 'min_impurity_decrease': [0.0, 0.1, 0.2], 0
    'class_weight': ['balanced', None],
    'ccp_alpha': [0.0, 0.01, 0.2]
}

# Perform GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(dt, param_grid)
grid_search.fit(X_train_scaled, y_train)


# Evaluate the best model on the test set
best_dt = grid_search.best_estimator_
test_score = best_dt.score(X_test_scaled, y_test)
print('SUCCESS')

# %% [markdown]
# **Model Evaluation after Tuning**

# %%
# Evaluate the model
best_dt.fit(X_train_scaled, y_train)
y_pred = best_dt.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

pickle.dump(best_dt, open(model_dump_path, 'wb'))
