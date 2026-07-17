import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


df = pd.read_csv("full_dataset_v2.csv")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Shape:", df.shape)

# =====================================================
# TARGET
# =====================================================

TARGET_MAG = 5.0

df["target"] = (
        df["magnitude"] >= TARGET_MAG
).astype(int)

print("\nClass Distribution")
print(df["target"].value_counts())

# =====================================================
# FEATURE GROUPS
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
advanced_temporal = [

    "rolling_mean_mag_5",
    "rolling_mean_mag_10",

    "rolling_max_mag_5",
    "rolling_max_mag_10",

    "rolling_std_mag_10",

    "event_count_30d",
    "event_count_90d",
    "event_count_365d",

    "moment_sum_30d",
    "moment_sum_90d",

    "energy_sum_30d",
    "energy_sum_90d"
]

features = (
        spatial_features
        + physics_features
        + temporal_features
        + advanced_temporal
)
X = df[features]

y = df["target"]

groups = df["event_id"]

print("\nNumber of Features:", X.shape[1])
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)


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

groups_train = groups.iloc[train_idx]

print("\nTrain Size:", len(X_train))
print("Test Size :", len(X_test))

# RANDOM FOREST

rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n")
print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

print(classification_report(y_test, rf_pred))

print("Accuracy          :", accuracy_score(y_test, rf_pred))
print("Balanced Accuracy :", balanced_accuracy_score(y_test, rf_pred))
print("Precision         :", precision_score(y_test, rf_pred))
print("Recall            :", recall_score(y_test, rf_pred))
print("F1                :", f1_score(y_test, rf_pred))
print("ROC-AUC           :", roc_auc_score(y_test, rf_prob))
print("PR-AUC            :", average_precision_score(y_test, rf_prob))


# XGBOOST

scale_pos_weight = (
        (y_train == 0).sum()
        /
        (y_train == 1).sum()
)

xgb = XGBClassifier(

    n_estimators=800,

    max_depth=6,

    min_child_weight=3,

    gamma=0.05,

    learning_rate=0.03,

    subsample=0.85,
    colsample_bytree=0.85,

    scale_pos_weight=scale_pos_weight,

    random_state=42,
    eval_metric="logloss"
)
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_prob = xgb.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

plt.figure(figsize=(7,6))

plt.plot(
    fpr_rf,
    tpr_rf,
    linewidth=2,
    label=f"RF (AUC={auc(fpr_rf,tpr_rf):.3f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    linewidth=2,
    label=f"XGB (AUC={auc(fpr_xgb,tpr_xgb):.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    '--',
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure7_ROC_Comparison.png",
    dpi=600
)

from sklearn.metrics import precision_recall_curve

plt.figure(figsize=(7,6))

p_rf,r_rf,_ = precision_recall_curve(
    y_test,
    rf_prob
)

p_xgb,r_xgb,_ = precision_recall_curve(
    y_test,
    xgb_prob
)

plt.plot(
    r_rf,
    p_rf,
    label="RF"
)

plt.plot(
    r_xgb,
    p_xgb,
    label="XGB"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure8_PR_Comparison.png",
    dpi=600
)

plt.show()



print("\n")
print("=" * 60)
print("XGBOOST")
print("=" * 60)

print(classification_report(y_test, xgb_pred))

print("Accuracy          :", accuracy_score(y_test, xgb_pred))
print("Balanced Accuracy :", balanced_accuracy_score(y_test, xgb_pred))
print("Precision         :", precision_score(y_test, xgb_pred))
print("Recall            :", recall_score(y_test, xgb_pred))
print("F1                :", f1_score(y_test, xgb_pred))
print("ROC-AUC           :", roc_auc_score(y_test, xgb_prob))
print("PR-AUC            :", average_precision_score(y_test, xgb_prob))


print("\n")
print("=" * 60)
print("5-FOLD GROUP CROSS VALIDATION")
print("=" * 60)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_cv = cross_val_score(
    rf,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

xgb_cv = cross_val_score(
    xgb,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\nRandom Forest CV ROC-AUC")
print("Mean :", rf_cv.mean())
print("Std  :", rf_cv.std())

print("\nXGBoost CV ROC-AUC")
print("Mean :", xgb_cv.mean())
print("Std  :", xgb_cv.std())
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

plt.figure(figsize=(7,6))

plt.plot(
    fpr_rf,
    tpr_rf,
    linewidth=2,
    label=f"RF (AUC={auc(fpr_rf,tpr_rf):.3f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    linewidth=2,
    label=f"XGB (AUC={auc(fpr_xgb,tpr_xgb):.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    '--',
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure7_ROC_Comparison.png",
    dpi=600
)

plt.show()


print("\nFinished Successfully")