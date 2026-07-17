import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("full_dataset.csv")

TARGET_MAG = 4.5

df["target"] = (
    df["magnitude"] >= TARGET_MAG
).astype(int)


drop_cols = [
    "magnitude",
    "event_id",
    "target"
]

X = df.drop(columns=drop_cols)

y = df["target"]

groups = df["event_id"]

print("="*60)
print("DATASET")
print("="*60)

print("Shape:", X.shape)
print("Positive:", y.sum())
print("Negative:", (y==0).sum())



imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


scaler = StandardScaler()

X = scaler.fit_transform(X)



gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]



neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

weight_for_0 = 1.0
weight_for_1 = neg / pos

class_weight = {
    0: weight_for_0,
    1: weight_for_1
}

print("\nClass Weight:", class_weight)


# DNN


model = Sequential()

model.add(Dense(
    128,
    activation="relu",
    input_shape=(X_train.shape[1],)
))

model.add(Dropout(0.30))

model.add(Dense(
    64,
    activation="relu"
))

model.add(Dropout(0.30))

model.add(Dense(
    32,
    activation="relu"
))

model.add(Dense(
    1,
    activation="sigmoid"
))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["AUC"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=100,
    batch_size=64,
    verbose=1,
    class_weight=class_weight,
    callbacks=[early_stop]
)



prob = model.predict(X_test).flatten()

pred = (prob >= 0.30).astype(int)

print("\n")
print("="*60)
print("DNN RESULTS")
print("="*60)

print(classification_report(
    y_test,
    pred
))

print(
    "Balanced Accuracy:",
    balanced_accuracy_score(y_test, pred)
)

print(
    "Precision:",
    precision_score(y_test, pred)
)

print(
    "Recall:",
    recall_score(y_test, pred)
)

print(
    "F1:",
    f1_score(y_test, pred)
)

print(
    "ROC-AUC:",
    roc_auc_score(y_test, prob)
)

print(
    "PR-AUC:",
    average_precision_score(y_test, prob)
)