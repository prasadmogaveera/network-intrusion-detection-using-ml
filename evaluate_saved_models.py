
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt",
]

LABELS = ["Normal", "Anomaly"]

# Load the existing test data
test_df = pd.read_csv(BASE_DIR / "binary_test.csv")
X_test = test_df[FEATURES]
y_test = test_df["Label"].astype(str).to_numpy()

if not set(np.unique(y_test)).issubset(set(LABELS)):
    raise ValueError("Unexpected labels found in binary_test.csv")

# Load the saved models. No model is retrained.
rf = joblib.load(BASE_DIR / "random_forest_binary.joblib")
xgb = joblib.load(BASE_DIR / "xgboost_binary.joblib")
ann = tf.keras.models.load_model(BASE_DIR / "ann_binary.keras")
ann_scaler = joblib.load(BASE_DIR / "ann_binary_scaler.joblib")

# Random Forest predicts the original text labels
rf_pred = np.asarray(rf.predict(X_test)).astype(str)

# XGBoost was trained with Normal=0 and Anomaly=1
xgb_raw = np.asarray(xgb.predict(X_test)).ravel()

if set(np.unique(xgb_raw)).issubset({0, 1}):
    xgb_pred = np.where(xgb_raw == 1, "Anomaly", "Normal")
else:
    xgb_pred = xgb_raw.astype(str)

# ANN needs the scaler fitted during its original training
X_scaled = ann_scaler.transform(X_test)
ann_prob = ann.predict(X_scaled, verbose=0).ravel()
ann_pred = np.where(ann_prob >= 0.5, "Anomaly", "Normal")

predictions = {
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "ANN": ann_pred,
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, pred) in zip(axes, predictions.items()):
    if not set(np.unique(pred)).issubset(set(LABELS)):
        raise ValueError(
            f"Unexpected prediction labels from {name}: "
            f"{np.unique(pred)}"
        )

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(
        y_test, pred, pos_label="Anomaly", zero_division=0
    )
    recall = recall_score(
        y_test, pred, pos_label="Anomaly", zero_division=0
    )
    f1 = f1_score(
        y_test, pred, pos_label="Anomaly", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Anomaly_Precision": precision,
        "Anomaly_Recall": recall,
        "Anomaly_F1": f1,
    })

    print(f"\n{'=' * 55}")
    print(name)
    print(f"{'=' * 55}")
    print(classification_report(
        y_test,
        pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, pred, labels=LABELS)
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=LABELS,
    ).plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig(
    BASE_DIR / "consistent_evaluation_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

results_df = pd.DataFrame(results)
results_df.to_csv(
    BASE_DIR / "consistent_evaluation_results.csv",
    index=False,
)

print("\nCONSISTENT EVALUATION RESULTS")
print(results_df.round(4).to_string(index=False))
print("\nSaved: consistent_evaluation_results.csv")
print("Saved: consistent_evaluation_confusion_matrices.png")
