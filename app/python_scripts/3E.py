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

tf.keras.utils.set_random_seed(42)

processed_dir = os.path.join("data", "processed")
final_dataset_path = os.path.join(
    processed_dir, 'FEData_For_Modelling.parquet')
minority_users_nn_pq = os.path.join(processed_dir, 'minority_users_nn.parquet')

models_dir = os.path.join("models", "models")
model_dump_path = os.path.join(models_dir, 'nn_model.pkl')


# Import Dataset

# Read the multi-part Parquet dataset
fds = dd.read_parquet(final_dataset_path)
fds['malicious'] = fds['malicious'].astype(int)

X = fds.drop(columns=['malicious']).compute()
y = fds['malicious'].compute()

# Split the fds into training and testing sets
X_train_w_name, X_test_w_name, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train_w_name.drop(
    columns=["user",]), X_test_w_name.drop(columns=["user",])


# Standardize features, using minmax and not standardscale -> got dif??
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# print(X_test.shape, X_train.shape)

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


history = model.fit(X_train, y_train,
                    epochs=150,
                    batch_size=64,
                    validation_split=0.2,
                    shuffle=True,
                    # callbacks = [early_stopping]
                    )

pickle.dump(model.weights, open(model_dump_path, 'wb'))
print('COMPLETED')
