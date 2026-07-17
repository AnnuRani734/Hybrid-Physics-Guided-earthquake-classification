import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

import shap
import matplotlib.pyplot as plt



df = pd.read_csv("full_dataset_v2.csv")

TARGET_MAG = 5.0

df["target"] = (
    df["magnitude"] >= TARGET_MAG
).astype(int)


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


scale_pos_weight = (
    (y_train == 0).sum()
    /
    (y_train == 1).sum()
)

model = XGBClassifier(

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

model.fit(X_train, y_train)



print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(model)

sample_size = min(1000, len(X_test))

X_sample = X_test.sample(
    sample_size,
    random_state=42
)

shap_values = explainer.shap_values(X_sample)


plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "Figure19_SHAP_Summary.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Saved: Figure19_SHAP_Summary.png")


importance = np.abs(shap_values).mean(axis=0)

imp_df = pd.DataFrame({

    "Feature": X_sample.columns,
    "MeanAbsSHAP": importance

})

imp_df = imp_df.sort_values(
    "MeanAbsSHAP",
    ascending=False
)

imp_df.to_csv(
    "shap_feature_importance.csv",
    index=False
)

print("Saved: shap_feature_importance.csv")



top20 = imp_df.head(20)

plt.figure(figsize=(8, 8))

plt.barh(
    top20["Feature"][::-1],
    top20["MeanAbsSHAP"][::-1]
)

plt.xlabel("Mean |SHAP Value|")

plt.tight_layout()

plt.savefig(
    "Figure20_SHAP_Top20.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

print("Saved: Figure20_SHAP_Top20.png")

print("\nTop 20 Features:\n")
print(top20)