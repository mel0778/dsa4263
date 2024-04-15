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
processed_dir = os.path.join("data", "processed")
final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')
minority_users_dt_pq = os.path.join(
    processed_dir, 'minority_users_dt_smote.parquet')

# Model
models_dir = os.path.join("models", "models")
model_dump_path = os.path.join(models_dir, 'dt_smote_model.pkl')


# %% [markdown]
# Import Dataset

# %%
# Read the multi-part Parquet dataset
fds = pd.read_parquet(final_dataset_path)
fds.head(4000)

# %% [markdown]
# # Decision Tree / Random Forest

# %%
X = fds.drop(columns=['user', 'malicious'])
y = fds.malicious

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Train Labels before Resampling")
print(Counter(y_train))

# %% [markdown]
# **SMOTE Oversampling**

# %%
# transform the dataset
# sampling_strategy=0.8
oversample = SMOTE(sampling_strategy=0.4, random_state=42)
resampled_X_train, resampled_y_train = oversample.fit_resample(
    X_train, y_train)

print("Train Labels after Resampling")
print(Counter(resampled_y_train))

# %% [markdown]
# **Feature Normalisation**

# %%
# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(resampled_X_train)
X_test_scaled = scaler.transform(X_test)

y_train = resampled_y_train

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

# Print the best hyperparameters and corresponding score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_dt = grid_search.best_estimator_
test_score = best_dt.score(X_test_scaled, y_test)
print("Test Set Score:", test_score)


# %% [markdown]
# **Model Evaluation after Tuning**

# %%
# Evaluate the model
best_dt.fit(X_train_scaled, y_train)
pickle.dump(best_dt, open(model_dump_path, 'wb'))
print('COMPLETED')
