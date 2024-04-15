# %%
# import necessary libraries
import pyarrow
import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import os
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve,
                             ConfusionMatrixDisplay, classification_report, auc, roc_curve, precision_recall_fscore_support, f1_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, regularizers, models
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve
import pickle


# %%
tf.keras.utils.set_random_seed(42)

# %% [markdown]
# # Paths

# %%
processed_dir = os.path.join("..", "data", "processed")
final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')
minority_users_nn_pq = os.path.join(
    processed_dir, 'minority_users_nn_smote.parquet')

models_dir = os.path.join("..", "models", "models")
model_dump_path = os.path.join(models_dir, 'nn_model_smote.pkl')

figures_dir = os.path.join("..", "materials", "reports", "figures", "3F")

nn_smote_loss_plot_path = os.path.join(figures_dir, 'nn_loss_plot.png')
nn_smote_acc_plot_path = os.path.join(figures_dir, 'nn_acc_plot.png')

nn_smote_confusion_matrix_path = os.path.join(
    figures_dir, 'nn_confusion_matrix.png')

nn_smote_train_confidence_score_path = os.path.join(
    figures_dir, 'nn_train_confidence_score.png')
nn_smote_test_confidence_score_path = os.path.join(
    figures_dir, 'nn_test_confidence_score.png')

# %% [markdown]
# Import Dataset

# %%
# Read the multi-part Parquet dataset
fds = dd.read_parquet(final_dataset_path)
fds['malicious'] = fds['malicious'].astype(int)
fds.head(4000)

# %%
X = fds.drop(columns=['malicious']).compute()
y = fds['malicious'].compute()

# %%
# Split the fds into training and testing sets
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
# sampling_strategy=0.8
oversample = SMOTE(sampling_strategy=0.4, random_state=42)
X_train, y_train = oversample.fit_resample(X_train, y_train)

print("Train Labels after Resampling")
print(Counter(y_train))

# %%
# Standardize features, using minmax and not standardscale -> got dif??
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(X_test.shape, X_train.shape)

# %% [markdown]
# # Hyperparameter Tuning

# %%
# Define X_train, y_train, X_test, y_test here
# Define the neural network architecture


def create_model(neurons_layer1=64, neurons_layer2=32, learning_rate=0.01):
    model = Sequential([
        Dense(neurons_layer1, activation='relu', input_shape=(17,)),
        Dense(neurons_layer2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['recall'])
    return model


# Define parameter values to loop over
neurons_layer1_values = [32, 64, 128]
neurons_layer2_values = [16, 32, 64]
learning_rates = [0.001, 0.01, 0.1]

best_recall = 0
best_params = {}

# Iterate over parameter combinations
for neurons_layer1 in neurons_layer1_values:
    for neurons_layer2 in neurons_layer2_values:
        for learning_rate in learning_rates:
            # Create the model
            model = create_model(neurons_layer1=neurons_layer1,
                                 neurons_layer2=neurons_layer2, learning_rate=learning_rate)

            # Train the model
            history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                                validation_split=0.2, shuffle=True, verbose=0)

            # Evaluate the model
            loss, recall = model.evaluate(X_test, y_test)

            # Check if current accuracy is better than the best accuracy so far
            if recall > best_recall:
                best_recall = recall
                best_params = {'neurons_layer1': neurons_layer1,
                               'neurons_layer2': neurons_layer2, 'learning_rate': learning_rate}

# Print best parameters and corresponding accuracy
print("Best parameters found: ", best_params)
print("Best Recall: ", best_recall)

# %% [markdown]
# ## Building the model

# %%
# Define the neural network architecture
model = Sequential([
    Dense(best_params['neurons_layer1'], activation='relu', input_shape=(17,)),
    Dense(best_params['neurons_layer2'], activation='relu'),
    Dense(1, activation='sigmoid')
])
optimizer = Adam(learning_rate=best_params['learning_rate'])
# Compile the model
model.compile(optimizer=optimizer,
              loss='binary_crossentropy', metrics=['recall'])


# %%
history = model.fit(X_train, y_train,
                    epochs=150,
                    batch_size=64,
                    validation_split=0.2,
                    shuffle=True,
                    # callbacks = [early_stopping]
                    )

# Evaluate the model
loss, recall = model.evaluate(X_train, y_train)
print("Final Loss:", loss)
print("Final Recall:", recall)

# %%
# loss plot
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(nn_smote_loss_plot_path)
plt.show()

# %%
# acc plot
plt.plot(history.history["recall"], label="Training Recall")
plt.plot(history.history["val_recall"], label="Validation Recall")
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(nn_smote_acc_plot_path)
plt.show()

# %% [markdown]
# Prediction

# %% [markdown]
# Threshold tuning

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


# %%
y_pred_prob = model.predict(X_test)
threshold = threshold_tuning(
    y_test, y_pred_prob, tp_weight=2, fp_weight=1, fn_weight=0.5, threshold=0.5)
print(threshold)
y_pred = (y_pred_prob > threshold).astype(int)
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
disp.plot(cmap=plt.cm.Greens)
plt.savefig(nn_smote_confusion_matrix_path)
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
    plt.plot(dt_fpr, dt_tpr, color='green', lw=2,
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.legend(loc="lower right")

    # plt.legend()


plot_roc_curve(y_test, y_pred)
print(f'AUC score: {roc_auc_score(y_test, y_pred)}')

# %% [markdown]
# ### Risk scores

# %%


def custom_min_max_scale(arr):
    # Identify positive and negative values
    positive_values = arr[arr > 0]
    negative_values = arr[arr < 0]

    # Scale positive values to [0, 1]
    positive_scaled = positive_values / np.max(positive_values)

    # Scale negative values to [-1, 0]
    negative_scaled = negative_values / np.min(negative_values)

    # Replace original array with scaled values
    arr[arr > 0] = positive_scaled
    arr[arr < 0] = negative_scaled

    return arr

# %%


def get_risk_score(X, threshold):
    prob = model.predict(X)
    data_array = threshold - prob
    confidence = abs(custom_min_max_scale(data_array))

    return confidence


# %%
train_risk = get_risk_score(X_train, threshold)
test_risk = get_risk_score(X_test, threshold)

# %%
# Plot histogram for train risk score
plt.yscale('log')
plt.hist(train_risk, bins=30, edgecolor='black',
         color="green")  # Adjust bins as needed
plt.xlabel('Train Confidence Scores')
plt.ylabel('Frequency')
plt.title('Neural Network Histogram of Train Data Confidence Scores (LogScaled)')
plt.grid(True)
plt.savefig(nn_smote_train_confidence_score_path)
plt.show()

# %%
# Plot histogram for test risk score
plt.yscale('log')
plt.hist(test_risk, bins=30, edgecolor='black',
         color="green")  # Adjust bins as needed
plt.xlabel('Test Confidence Scores')
plt.ylabel('Frequency')
plt.title('Neural Network Histogram of Test Data Confidence Scores (LogScaled)')
plt.grid(True)
plt.savefig(nn_smote_test_confidence_score_path)
plt.show()


# %%
print("Example Train Risk Scores:")
print(train_risk[:10])
print("\nExample Test Risk Scores:")
print(test_risk[:10])

# %% [markdown]
# # Export Model

# %%
pickle.dump(model.weights, open(model_dump_path, 'wb'))

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
names_array = list(X_test_w_name['user'])
final_guess = {'User': names_array, 'Actual': list(y_test.astype(bool)), 'Prediction': list(
    y_pred.flatten().astype(bool)), 'Confidence of Prediction': list(test_risk.flatten())}
refactored_df = pd.DataFrame(final_guess)
df_toparquet(refactored_df, minority_users_nn_pq)
refactored_df


# %%
print("False Positives")
get_minority_tables(1, 0, refactored_df)

print("False Negatives")
get_minority_tables(0, 1, refactored_df)

print("True Positives")
get_minority_tables(1, 1, refactored_df)
