# %%
import dask.dataframe as dd
import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve,
                             ConfusionMatrixDisplay, classification_report, auc, roc_curve, precision_recall_fscore_support, f1_score)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from itertools import permutations
import pandas as pd

# %% [markdown]
# # Paths

# %%
processed_dir = os.path.join("..", "data", "processed")

final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')

minority_users_dt_pq = os.path.join(processed_dir, 'minority_users_dt.parquet')
minority_users_smote_dt_pq = os.path.join(
    processed_dir, 'minority_users_dt_smote.parquet')

minority_users_svm_pq = os.path.join(
    processed_dir, 'minority_users_svm.parquet')
minority_users_smote_svm_pq = os.path.join(
    processed_dir, 'minority_users_svm_smote.parquet')

minority_users_nn_pq = os.path.join(processed_dir, 'minority_users_nn.parquet')
minority_users_smote_nn_pq = os.path.join(
    processed_dir, 'minority_users_nn_smote.parquet')

minority_users_em_pq = os.path.join(
    processed_dir, 'minority_users_em_pq.parquet')


models_dir = os.path.join("..", "models", "models")
model_dump_path = os.path.join(models_dir, 'em_model.pkl')

# Import Dataset

# Read the multi-part Parquet dataset
dt_data = dd.read_parquet(minority_users_dt_pq)
smote_dt_data = dd.read_parquet(minority_users_smote_dt_pq)

svm_data = dd.read_parquet(minority_users_svm_pq)
smote_svm_data = dd.read_parquet(minority_users_smote_svm_pq)

nn_data = dd.read_parquet(minority_users_nn_pq)
smote_nn_data = dd.read_parquet(minority_users_smote_nn_pq)


# Checking if all Test users are there and are the same

# %%
a = set(svm_data['User'].values.compute())
b = set(nn_data['User'].values.compute())
c = set(dt_data['user'].values.compute())
d = set(smote_svm_data['User'].values.compute())
e = set(smote_nn_data['User'].values.compute())
f = set(smote_dt_data['user'].values.compute())
if a == b == c == d == e == f:
    print("All variables are equal.")


# %%
# Sort dt_data by 'User' column
sorted_dt_data = dt_data.sort_values(by='user')

# Sort smote_dt_data by 'User' column
sorted_smote_dt_data = smote_dt_data.sort_values(by='user')

# Sort svm_data by 'User' column
sorted_svm_data = svm_data.sort_values(by='User')

# Sort smote_svm_data by 'User' column
sorted_smote_svm_data = smote_svm_data.sort_values(by='User')

# Sort nn_data by 'User' column
sorted_nn_data = nn_data.sort_values(by='User')

# Sort smote_nn_data by 'User' column
sorted_smote_nn_data = smote_nn_data.sort_values(by='User')


# Fixing DT exported dataframes before Ensembling

def fix_dt_df(df):
    df['Confidence of Prediction'] = df[['normal_prob', 'malicious_prob']].apply(
        lambda row: max(row), axis=1, meta=('float'))
    df = df.drop(['Decision Path', 'GINI Confidence of Prediction',
                 'normal_prob', 'malicious_prob'], axis=1)
    return df


sorted_dt_data = fix_dt_df(sorted_dt_data)
sorted_smote_dt_data = fix_dt_df(sorted_smote_dt_data)
sorted_smote_dt_data

# Threshold Tuning

# %%


def calculate_tp_fp_fn(y_true, y_pred_prob, threshold):
    """
    Calculate True Positives (TP), False Positives (FP), and False Negatives (FN) 
    based on the given true labels (y_true) and predicted labels (y_pred) with a given threshold.

    Args:
    - y_true: True labels
    - y_pred: Predicted labels (probabilities or scores)
    - threshold: Threshold for classification

    Returns:
    - TP: Number of True Positives
    - FP: Number of False Positives
    - FN: Number of False Negatives
    """
    y_pred_binary = (y_pred_prob > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    TN, FP, FN, TP = cm.ravel()

    return TP, FP, FN


def threshold_tuning(y_test, y_pred_prob, tp_weight, fp_weight, fn_weight, threshold=0.5):
    best_t_score = float('-inf')
    best_threshold = threshold
    while threshold >= 0:
        Tp, Fp, Fn = calculate_tp_fp_fn(y_test, y_pred_prob, threshold)
        t_score = Tp*tp_weight - Fp*fp_weight - Fn*fn_weight
        if t_score > best_t_score:
            best_t_score = t_score
            best_threshold = threshold
        threshold -= 0.01
    return best_threshold


def ensemble_predictions(*models):
    # Extract confidence values and predictions from each model
    confidences = []
    predictions = []
    for model in models:
        confidences.append(model['Confidence of Prediction'].values.compute())
        predictions.append(model['Prediction'].values.compute())

    # Calculate total confidence across all models
    total_conf = sum(confidences)
    conf_score = sum(confidences*(confidences/total_conf))
    # Calculate weights for each model
    weights = [conf / total_conf for conf in confidences]

    # Calculate weighted predictions
    y_pred = sum(weight * pred for weight, pred in zip(weights, predictions))

    return y_pred, conf_score


def find_best_ensemble_permutation(y_true, *models, threshold=1, tp_weight=2, fp_weight=1, fn_weight=0.5):
    best_score = float('-inf')
    best_permutation = None
    best_y_pred_prob = None
    best_conf_score = None
    total_permutations = len(list(permutations(models)))
    print(total_permutations)
    for permutation in permutations(models):
        y_pred_prob, conf_score = ensemble_predictions(*permutation)
        tp, fp, fn = calculate_tp_fp_fn(y_true, y_pred_prob, threshold)
        score = tp * tp_weight
        if score > best_score:
            best_score = score
            best_permutation = permutation
            best_y_pred_prob = y_pred_prob
            best_conf_score = conf_score
        total_permutations -= 1
        print(total_permutations)
    return best_permutation, best_score, best_y_pred_prob, best_conf_score


y_test = sorted_svm_data['Actual'].values.compute()
best_permutation, best_score, best_y_pred_prob, best_conf_score = find_best_ensemble_permutation(
    y_test, sorted_svm_data, sorted_nn_data, sorted_dt_data, sorted_smote_svm_data, sorted_smote_nn_data, sorted_smote_dt_data, threshold=0)

# Threshold tuning
threshold = threshold_tuning(
    y_test, best_y_pred_prob, tp_weight=2, fp_weight=1, fn_weight=0.5, threshold=1)

y_pred = (best_y_pred_prob > threshold).astype(int)
model_weights = [best_permutation, threshold]
pickle.dump(model.weights, open(model_dump_path, 'wb'))
