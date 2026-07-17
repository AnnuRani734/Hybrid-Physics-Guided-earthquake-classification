import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("full_dataset.csv")

TARGET_MAG = 5.0

df["target"] = (
    df["magnitude"] >= TARGET_MAG
).astype(int)

# =====================================================
# FEATURES
# =====================================================

spatial_features = [
    "latitude",
    "longitude",
    "depth_km"
]

physics_features = [

    "H_m",
    "B",

    "c_phase_1Hz",
    "c_phase_2Hz",
    "c_phase_5Hz",

    "c_group_12",
    "c_group_25",
    "c_group_mean",

    "delta_cpB_1Hz",
    "delta_cpB_2Hz",
    "delta_cpB_5Hz",

    "interface_damage_index",

    "disp_slope_12",
    "disp_slope_25",

    "disp_curvature",

    "dispersion_range",

    "gp_ratio_12",
    "gp_ratio_25",

    "idi_norm",

    "freq_sensitivity"
]

temporal_features = [

    "mag_t-1",
    "mag_t-2",
    "mag_t-3",
    "mag_t-4",
    "mag_t-5",
    "mag_t-6",

    "dt_t-1",
    "dt_t-2",
    "dt_t-3",
    "dt_t-4",
    "dt_t-5",
    "dt_t-6"
]

features = (
    spatial_features
    + physics_features
    + temporal_features
)

X = df[features]
y = df["target"]
groups = df["event_id"]

# =====================================================
# IMPUTATION
# =====================================================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# =====================================================
# GROUP SPLIT
# =====================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_prob = rf.predict_proba(X_test)[:, 1]

# =====================================================
# XGBOOST
# =====================================================

scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    min_child_weight=5,
    gamma=0.1,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_prob = xgb.predict_proba(X_test)[:, 1]

# =====================================================
# THRESHOLD STUDY
# =====================================================

thresholds = [0.50, 0.40, 0.30, 0.20]

results = []

print("\n")
print("="*70)
print("RANDOM FOREST THRESHOLD STUDY")
print("="*70)

for th in thresholds:

    pred = (rf_prob >= th).astype(int)

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(
        y_test,
        pred
    )

    print(
        f"Threshold={th:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f} | "
        f"BalancedAcc={bal_acc:.3f}"
    )

    results.append([
        "RF",
        th,
        precision,
        recall,
        f1,
        bal_acc
    ])

print("\n")
print("="*70)
print("XGBOOST THRESHOLD STUDY")
print("="*70)

for th in thresholds:

    pred = (xgb_prob >= th).astype(int)

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(
        y_test,
        pred
    )

    print(
        f"Threshold={th:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f} | "
        f"BalancedAcc={bal_acc:.3f}"
    )

    results.append([
        "XGB",
        th,
        precision,
        recall,
        f1,
        bal_acc
    ])

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "Balanced_Accuracy"
    ]
)

results_df.to_csv(
    "threshold_study_results.csv",
    index=False
)

print("\n")
print(results_df)

print("\nSaved: threshold_study_results.csv")

print(
    "\nROC-AUC RF  :",
    roc_auc_score(y_test, rf_prob)
)

print(
    "PR-AUC RF   :",
    average_precision_score(y_test, rf_prob)
)

print(
    "\nROC-AUC XGB :",
    roc_auc_score(y_test, xgb_prob)
)

print(
    "PR-AUC XGB  :",
    average_precision_score(y_test, xgb_prob)
)