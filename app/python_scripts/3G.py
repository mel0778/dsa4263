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


# Figures
figures_dir = os.path.join("..", "materials", "reports", "figures", "3G")
em_confusion_matrix_path = os.path.join(figures_dir, 'em_confusion_matrix.png')
em_test_confidence_score_path = os.path.join(
    figures_dir, 'em_test_confidence_score.png')

# %% [markdown]
# Import Dataset

# %%
# Read the multi-part Parquet dataset
dt_data = dd.read_parquet(minority_users_dt_pq)
smote_dt_data = dd.read_parquet(minority_users_smote_dt_pq)

svm_data = dd.read_parquet(minority_users_svm_pq)
smote_svm_data = dd.read_parquet(minority_users_smote_svm_pq)

nn_data = dd.read_parquet(minority_users_nn_pq)
smote_nn_data = dd.read_parquet(minority_users_smote_nn_pq)

svm_data.head(4000)

# %% [markdown]
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

# %% [markdown]
# Sorting Alphabetically

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

sorted_dt_data.head()

# %% [markdown]
# Fixing DT exported dataframes before Ensembling

# %%


def fix_dt_df(df):
    df['Confidence of Prediction'] = df[['normal_prob', 'malicious_prob']].apply(
        lambda row: max(row), axis=1, meta=('float'))
    df = df.drop(['Decision Path', 'GINI Confidence of Prediction',
                 'normal_prob', 'malicious_prob'], axis=1)
    return df


sorted_dt_data = fix_dt_df(sorted_dt_data)
sorted_smote_dt_data = fix_dt_df(sorted_smote_dt_data)
sorted_smote_dt_data

# %% [markdown]
# # Selecting Best Ensemble

# %%
'''def ensemble_predictions(dt,svm,nn):
    conf_dt = np.maximum(dt['malicious_prob'].values.compute(),dt['normal_prob'].values.compute()) 
    conf_svm = svm['Confidence of Prediction'].values.compute()
    conf_nn = nn['Confidence of Prediction'].values.compute()
    total_conf = conf_dt + conf_svm + conf_nn
    weight_dt = conf_dt/total_conf
    weight_svm = conf_svm/total_conf
    weight_nn = conf_nn/total_conf
    y_pred = weight_dt*dt['Prediction'].values.compute() + weight_svm*svm['Prediction'].values.compute() + weight_nn*nn['Prediction'].values.compute()
    return  y_pred
    #dt[malicious_prob]*'''

# %% [markdown]
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

# %%


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

# %% [markdown]
# Ensemble Selection Tuning

# %%


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

# %%


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


# %%
y_test = sorted_svm_data['Actual'].values.compute()
best_permutation, best_score, best_y_pred_prob, best_conf_score = find_best_ensemble_permutation(
    y_test, sorted_svm_data, sorted_nn_data, sorted_dt_data, sorted_smote_svm_data, sorted_smote_nn_data, sorted_smote_dt_data, threshold=0)
print("Best permutation:", best_permutation)
print("Best score:", best_score)

# %% [markdown]
# Threshold tuning

# %%
threshold = threshold_tuning(
    y_test, best_y_pred_prob, tp_weight=2, fp_weight=1, fn_weight=0.5, threshold=1)
print(threshold)
y_pred = (best_y_pred_prob > threshold).astype(int)
set(list(y_pred.flatten()))


accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=['normal', 'malicious'])

print("Accuracy = {:.2f}%".format(accuracy_score(y_test, y_pred)*100))
print("Precision = {}".format(precision_score(y_test, y_pred)))
print("Recall = {}".format(recall_score(y_test, y_pred)))
print("f-1 score = {}".format(f1_score(y_test, y_pred)))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
disp.plot(cmap=plt.cm.Reds)
plt.title('Ensemble Model Confusion Matrix')
plt.savefig(em_confusion_matrix_path)
plt.show()

# %%
# ROC - AUC


def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """
    plt.figure(figsize=(8, 6))
    # y_prob = best_dt.predict_proba(X_test_scaled)[::,1]
    dt_fpr, dt_tpr, thresholds = roc_curve(true_y, y_prob)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_auc = auc(dt_fpr, dt_tpr)
    plt.plot(dt_fpr, dt_tpr, color='red', lw=2,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.legend(loc="lower right")

    # plt.legend()


plot_roc_curve(y_test, y_pred)
print(f'AUC score: {roc_auc_score(y_test, y_pred)}')

# %%
# Plot histogram for test risk score
plt.hist(best_conf_score, bins=30, edgecolor='black',
         color="red")  # Adjust bins as needed
plt.xlabel('Test Confidence Scores')
plt.ylabel('Frequency')
plt.title('Ensemble Model Histogram of Test Data Confidence Scores')
plt.grid(True)
plt.savefig(em_test_confidence_score_path)
plt.show()

# %% [markdown]
# #  Our Prediction and the Confidence Associated with it

# %% [markdown]
# ## Helper Functions

# %%


def df_toparquet(pdf, path):
    ddf = dd.from_pandas(pdf)
    # Export the DataFrame to a parquet file=
    ddf.to_parquet(path, engine='pyarrow')


# %%
def get_minority_tables(y_pred_value, y_test_value, df):
    # Get rows of minority data
    minority_data = df[(df['Actual'] == y_test_value) &
                       (df['Prediction'] == y_pred_value)]
    display(minority_data)

# %% [markdown]
# Creating Table for export


# %%
names_array = list(smote_svm_data['User'].values.compute())
final_guess = {'User': names_array, 'Actual': list(y_test.astype(bool)), 'Prediction': list(
    y_pred.flatten().astype(bool)), 'Confidence of Prediction': list(best_conf_score.flatten())}
refactored_df = pd.DataFrame(final_guess)
df_toparquet(refactored_df, minority_users_em_pq)
refactored_df


# %%
print("False Positives")
get_minority_tables(1, 0, refactored_df)

print("False Negatives")
get_minority_tables(0, 1, refactored_df)

print("True Positives")
get_minority_tables(1, 1, refactored_df)
