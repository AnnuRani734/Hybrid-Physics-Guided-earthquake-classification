import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt



df = pd.read_csv("full_dataset_v2.csv")



exclude_cols = [
    "magnitude",
    "event_id"
]

features = [
    c for c in df.columns
    if c not in exclude_cols
]

X = df[features]

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
    gss.split(X, np.zeros(len(X)), groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]


# THRESHOLDS
thresholds = [4.0, 4.5, 5.0]

results = []


for THRESHOLD in thresholds:

    print("\n" + "="*60)
    print(f"THRESHOLD = {THRESHOLD}")
    print("="*60)

    y = (
        df["magnitude"] >= THRESHOLD
    ).astype(int)

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    positives = y.sum()
    negatives = len(y) - positives

    print(
        f"Positive={positives}, "
        f"Negative={negatives}"
    )

    scale_pos_weight = (
        (y_train == 0).sum()
        /
        max((y_train == 1).sum(), 1)
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

    model.fit(
        X_train,
        y_train
    )

    prob = model.predict_proba(X_test)[:,1]

    pred = (
        prob >= 0.50
    ).astype(int)

    roc = roc_auc_score(
        y_test,
        prob
    )

    pr = average_precision_score(
        y_test,
        prob
    )

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

    bal = balanced_accuracy_score(
        y_test,
        pred
    )

    results.append({

        "Threshold": THRESHOLD,

        "ROC_AUC": roc,
        "PR_AUC": pr,

        "Precision": precision,
        "Recall": recall,
        "F1": f1,

        "Balanced_Accuracy": bal
    })


results_df = pd.DataFrame(results)

print("\n")
print(results_df)

results_df.to_csv(
    "threshold_study_results.csv",
    index=False
)

print(
    "\nSaved: threshold_study_results.csv"
)

plt.figure(figsize=(8,5))

plt.plot(
    results_df["Threshold"],
    results_df["ROC_AUC"],
    marker="o",
    linewidth=2,
    label="ROC-AUC"
)

plt.plot(
    results_df["Threshold"],
    results_df["PR_AUC"],
    marker="s",
    linewidth=2,
    label="PR-AUC"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="^",
    linewidth=2,
    label="Recall"
)

plt.xlabel(
    "Magnitude Threshold"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Threshold Sensitivity Analysis"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Figure21_Threshold_Sensitivity.png",
    dpi=600
)

plt.show()

print(
    "Saved: Figure21_Threshold_Sensitivity.png"
)
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("physics_dataset.csv")

plt.figure(figsize=(8,5))

plt.hist(
    df["magnitude"],
    bins=25,
    edgecolor="black"
)

plt.xlabel("Magnitude")
plt.ylabel("Count")
plt.title("Magnitude Distribution")

plt.tight_layout()

plt.savefig(
    "Figure2_Magnitude_Distribution.png",
    dpi=600
)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("physics_dataset.csv")

plt.figure(figsize=(8,6))

sc = plt.scatter(
    df["longitude"],
    df["latitude"],
    c=df["magnitude"],
    s=10
)

plt.colorbar(
    sc,
    label="Magnitude"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title(
    "Spatial Distribution of Earthquakes"
)

plt.tight_layout()

plt.savefig(
    "Figure3_Spatial_Distribution.png",
    dpi=600
)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("physics_dataset.csv")

df["time"] = pd.to_datetime(df["time"])

df["year"] = df["time"].dt.year

yearly = df.groupby("year").size()

plt.figure(figsize=(10,5))

plt.plot(
    yearly.index,
    yearly.values,
    marker="o"
)

plt.xlabel("Year")
plt.ylabel("Number of Events")

plt.title(
    "Annual Earthquake Counts"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "Figure4_Annual_Seismicity.png",
    dpi=600
)
plt.show()
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("full_dataset_v2.csv")


df["Class"] = df["magnitude"].apply(
    lambda x: "M≥5" if x >= 5.0 else "M<5"
)


physics_features = [

    "c_phase_1Hz",
    "c_phase_2Hz",

    "c_group_mean",

    "delta_cpB_2Hz"

]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,8)
)

axes = axes.flatten()

for i, feature in enumerate(physics_features):

    df.boxplot(
        column=feature,
        by="Class",
        ax=axes[i]
    )

    axes[i].set_title(feature)

plt.suptitle(
    "Physics Features by Magnitude Class"
)

plt.tight_layout()

plt.savefig(
    "Figure11_Physics_Boxplots.png",
    dpi=600
)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("full_dataset_v2.csv")


df["Class"] = df["magnitude"].apply(
    lambda x: "M≥5" if x >= 5.0 else "M<5"
)


features = [

    "event_count_90d",

    "event_count_365d",

    "rolling_mean_mag_10",

    "moment_sum_90d"

]



fig, axes = plt.subplots(
    2,
    2,
    figsize=(12,8)
)

axes = axes.flatten()

for i, feature in enumerate(features):

    df.boxplot(
        column=feature,
        by="Class",
        ax=axes[i]
    )

    axes[i].set_title(feature)

plt.suptitle(
    "Temporal Features by Magnitude Class"
)

plt.tight_layout()

plt.savefig(
    "Figure12_Temporal_Boxplots.png",
    dpi=600
)

plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("full_dataset_v2.csv")

top_features = [

    "event_count_90d",
    "event_count_365d",

    "rolling_mean_mag_10",
    "rolling_mean_mag_5",

    "moment_sum_90d",

    "mag_t-1",
    "mag_t-2",
    "mag_t-3",

    "depth_km",

    "H_m",

    "latitude",
    "longitude"
]

corr = df[top_features].corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Matrix of Key Predictors"
)

plt.tight_layout()

plt.savefig(
    "Figure13_Correlation_Heatmap.png",
    dpi=600
)

plt.show()
import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame({

    "Model":[
        "RF",
        "XGB",
        "DNN"
    ],

    "ROC_AUC":[
        0.697,
        0.670,
        0.690
    ],

    "PR_AUC":[
        0.299,
        0.272,
        0.258
    ]
})

ax = results.plot(
    x="Model",
    kind="bar",
    figsize=(8,5)
)

plt.title(
    "Model Performance Comparison"
)

plt.ylabel("Score")

plt.tight_layout()

plt.savefig(
    "Figure14_Model_Comparison.png",
    dpi=600
)

plt.show()