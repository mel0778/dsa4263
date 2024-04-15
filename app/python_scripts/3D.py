# %% [markdown]
# ## Import libs and data

# %%
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

# %% [markdown]
# # Paths

# %%
processed_dir = os.path.join("..", "data", "processed")
final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')
minority_users_svm_pq = os.path.join(
    processed_dir, 'minority_users_svm_smote.parquet')

figures_dir = figures_dir = os.path.join(
    "..", "materials", "reports", "figures", "3D")

models_dir = os.path.join("..", "models", "models")
model_dump_path = os.path.join(models_dir, 'svm_model_smote.pkl')

# %% [markdown]
# Import Dataset

# %%
# Read the multi-part Parquet dataset
fds = dd.read_parquet(final_dataset_path)
fds.head(4000)

# %%
y = fds["malicious"].compute()
X = fds.drop(columns=["malicious",]).compute()
# X = fds.drop(columns=["targetLabel","user"]).compute()

# %% [markdown]
# ## Preprocessing and Split

# %%
X_train_w_name, X_test_w_name, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train_w_name.drop(
    columns=["user",]), X_test_w_name.drop(columns=["user",])
print("Train Labels before Resampling")
print(Counter(y_train))

# %% [markdown]
# ## SMOTE

# %%
# transform the dataset
oversample = SMOTE(sampling_strategy=0.4, random_state=42)
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]


print("Train Labels after Resampling")
print(Counter(y_train))

# %% [markdown]
# ## Scaler

# %%
# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## SVM

# %%


def make_binary_labels(labels):
    return np.where(labels == 1, -1, 1)


y_train_gs = make_binary_labels(y_train)
y_test_gs = make_binary_labels(y_test)


# %%
# Create a scorer using make_scorer
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

# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


# %%
best_clf = grid

# %%
# (-1 for outliers, 1 for inliers)
print("##############################")
print("Test Evaluation")
print("##############################")

test_pred = best_clf.predict(X_test_scaled)

print(classification_report(y_true=y_test_gs, y_pred=test_pred,
      labels=[1, -1], target_names=['Normal', 'Malicious']))

conf_matrix = confusion_matrix(y_test_gs, test_pred, labels=[1, -1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, display_labels=['Normal', 'Malicious'])
disp.plot(cmap=plt.cm.Blues)
plt.title('SVM Confusion Matrix (With SMOTE)')
plt_path = os.path.join(figures_dir, "svm_smote_confusion_matrix.png")
plt.savefig(plt_path)
plt.show()

# %%


def plot_roc(clf, y_test, y_pred):

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=-1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for One-Class SVM')
    plt.legend(loc="lower right")
    plt.show()


# %%
# ROC for best SVM, grid search
plot_roc(best_clf, y_test_gs, test_pred)

# %% [markdown]
# ## Confidence scores

# %%


class ConfidenceScore:
    def __init__(self, clf, X):
        self.clf = clf
        distances = clf.decision_function(X)
        self.min = min(distances)
        self.max = max(distances)

    def get_confi_score(self, X):
        def min_max_scaling(value, new_min, new_max):
            old_min = min(min(value), self.min)
            old_max = max(max(value), self.max)
            if old_max == old_min:
                return np.zeros_like(value) if old_max == 0 else np.ones_like(value) * ((new_max - new_min) / 2) + new_min
            else:
                scaled_value = ((value - old_min) / (old_max -
                                old_min)) * (new_max - new_min) + new_min
                return scaled_value

        # Distance from decision boundary where positive = non-anomaly and negative = anomaly
        distances = self.clf.decision_function(X)

        # Scale negative distances on the negative scale and positive distances on the positive scale
        neg_distances = distances[distances < 0]
        pos_distances = distances[distances >= 0]

        scaled_neg = min_max_scaling(neg_distances, -1, 0)
        scaled_pos = min_max_scaling(pos_distances, 0, 1)

        # Combine scaled distances
        risk_scores = np.empty_like(distances)
        risk_scores[distances < 0] = scaled_neg
        risk_scores[distances >= 0] = scaled_pos
        confidence = np.abs(risk_scores)
        return confidence


# %%
confidence_scorer = ConfidenceScore(best_clf, X_train_scaled)
train_confi = confidence_scorer.get_confi_score(X_train_scaled)
test_confi = confidence_scorer.get_confi_score(X_test_scaled)

# %%
# Plot histogram
plt_path = os.path.join(figures_dir, "svm_smote_train_confidence_score.png")
plt.hist(train_confi, bins=30, edgecolor='black')
plt.xlabel('Train Risk Scores')
plt.ylabel('Frequency')
plt.title('SVM(SMOTE) Histogram of Train Data Confidence Scores (With SMOTE)')
plt.grid(True)
plt.savefig(plt_path)
plt.show()

# %%

# Plot histogram
plt_path = os.path.join(figures_dir, "svm_smote_test_confidence_score.png")
plt.hist(test_confi, bins=30, edgecolor='black')
plt.xlabel('Test Risk Scores')
plt.ylabel('Frequency')
plt.title('SVM(SMOTE) Histogram of Test Data Confidence Scores (With SMOTE)')
plt.grid(True)
plt.savefig(plt_path)
plt.show()

# %% [markdown]
# # Export Model

# %%
pickle.dump(best_clf, open(model_dump_path, 'wb'))

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


# %%
names_array = list(X_test_w_name['user'].iloc)
test_pred_ = np.where(test_pred == 1, 0, 1)
final_guess = {'User': names_array, 'Actual': y_test, 'Prediction': test_pred_.astype(
    bool), 'Confidence of Prediction': test_confi}
final_guess = pd.DataFrame(final_guess)
df_toparquet(final_guess, minority_users_svm_pq)
final_guess


# %%
print("False Positives")
get_minority_tables(1, 0, final_guess)

print("False Negatives")
get_minority_tables(0, 1, final_guess)

print("True Positives")
get_minority_tables(1, 1, final_guess)
