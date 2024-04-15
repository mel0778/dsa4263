import dask.dataframe as dd
import os
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc, make_scorer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pickle

processed_dir = os.path.join("data", "processed")
final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')
minority_users_svm_pq = os.path.join(
    processed_dir, 'minority_users_svm.parquet')

models_dir = os.path.join("models", "models")
model_dump_path = os.path.join(models_dir, 'svm_model.pkl')

# Read the multi-part Parquet dataset
fds = dd.read_parquet(final_dataset_path)

y = fds["malicious"].compute()
X = fds.drop(columns=["malicious",]).compute()


X_train_w_name, X_test_w_name, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train_w_name, y_train = X_train_w_name[y_train == 0], y_train[y_train == 0]
X_train, X_test = X_train_w_name.drop(
    columns=["user"]), X_test_w_name.drop(columns=["user",])

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def make_binary_labels(labels):
    return np.where(labels == 1, -1, 1)


y_train_gs = make_binary_labels(y_train)
y_test_gs = make_binary_labels(y_test)


def custom_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=1)


scorer = make_scorer(custom_f1_score, greater_is_better=True)

# defining parameter range
param_grid = {
    # Regularization parameter
    'nu': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
    # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001, 0.0001, 1.0, 10.0]
}

grid = GridSearchCV(OneClassSVM(), param_grid,
                    refit=True, verbose=3, scoring=scorer)

# fitting the model for grid search
grid.fit(X_train_scaled, y_train_gs)

best_clf = grid

pickle.dump(best_clf, open(model_dump_path, 'wb'))
print('COMPLETED')
