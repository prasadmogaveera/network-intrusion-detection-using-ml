
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Protocol",
    "Src_Port",
    "Dst_Port",
    "Flow_Duration",
    "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts",
    "Pkt_Len_Min",
    "Pkt_Len_Max",
    "Pkt_Len_Mean",
    "Pkt_Len_Std",
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "SYN_Flag_Cnt",
    "ACK_Flag_Cnt",
    "RST_Flag_Cnt",
    "FIN_Flag_Cnt",
    "PSH_Flag_Cnt",
]

TARGET = "Label"

print("=" * 60)
print("XGBOOST - BINARY CLASSIFICATION")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(BASE_DIR / "binary_train.csv")
test_df = pd.read_csv(BASE_DIR / "binary_test.csv")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", len(FEATURES))

print("\nTraining class distribution:")
print(y_train.value_counts())

# --------------------------------------------------
# 3. ENCODE TARGET AND HANDLE IMBALANCE
# --------------------------------------------------

# XGBoost binary classification expects numeric targets.
# Normal = 0, Anomaly = 1.
y_train_encoded = (y_train == "Anomaly").astype(int)
y_test_encoded = (y_test == "Anomaly").astype(int)

normal_count = int((y_train == "Normal").sum())
anomaly_count = int((y_train == "Anomaly").sum())

scale_pos_weight = anomaly_count / normal_count

print("\nScale positive weight:", round(scale_pos_weight, 4))

# --------------------------------------------------
# 4. CREATE MODEL
# --------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

# --------------------------------------------------
# 5. TRAIN
# --------------------------------------------------

print("\nTraining started...")
model.fit(X_train, y_train_encoded)
print("Training completed!")

# --------------------------------------------------
# 6. TRAINING VS TEST EVALUATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING VS TEST PERFORMANCE")
print("=" * 60)

train_pred_encoded = model.predict(X_train)
test_pred_encoded = model.predict(X_test)

train_accuracy = accuracy_score(
    y_train_encoded, train_pred_encoded
)
test_accuracy = accuracy_score(
    y_test_encoded, test_pred_encoded
)

print(f"\nTraining accuracy: {train_accuracy:.4f} "
      f"({train_accuracy * 100:.2f}%)")

print(f"Test accuracy:     {test_accuracy:.4f} "
      f"({test_accuracy * 100:.2f}%)")

print(f"Accuracy gap:      "
      f"{(train_accuracy - test_accuracy) * 100:.2f} percentage points")

print("\nTraining classification report:")
print(
    classification_report(
        y_train_encoded,
        train_pred_encoded,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

print("\nTest classification report:")
print(
    classification_report(
        y_test_encoded,
        test_pred_encoded,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test_encoded,
    test_pred_encoded,
    labels=[0, 1],
)

print("\nTest confusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"],
)

disp.plot(cmap="Blues", values_format="d")
plt.title("XGBoost - Binary Classification")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "xgboost_binary_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# --------------------------------------------------
# 8. FEATURE IMPORTANCE
# --------------------------------------------------

print("\nTop 10 feature importances:")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(importance.head(10).to_string())

# --------------------------------------------------
# 9. SAVE MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(
    model,
    BASE_DIR / "xgboost_binary.joblib",
)

pd.DataFrame({
    "Metric": ["Training Accuracy", "Test Accuracy"],
    "Score": [train_accuracy, test_accuracy],
}).to_csv(
    BASE_DIR / "xgboost_binary_results.csv",
    index=False,
)

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)
print("Model: xgboost_binary.joblib")
print("Confusion matrix: xgboost_binary_confusion_matrix.png")
print("Results: xgboost_binary_results.csv")

print("\nXGBoost evaluation completed.")
