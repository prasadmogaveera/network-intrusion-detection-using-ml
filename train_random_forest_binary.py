
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# 1. PROJECT PATH AND CONFIGURATION
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

# --------------------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# --------------------------------------------------

print("=" * 60)
print("RANDOM FOREST - BINARY CLASSIFICATION")
print("=" * 60)

print("\nLoading datasets...")

train_df = pd.read_csv(BASE_DIR / "binary_train.csv")
test_df = pd.read_csv(BASE_DIR / "binary_test.csv")

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

print("\nDataset information")
print("-" * 40)
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", len(FEATURES))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# --------------------------------------------------
# 3. CREATE RANDOM FOREST MODEL
# --------------------------------------------------

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# --------------------------------------------------
# 4. TRAIN THE MODEL
# --------------------------------------------------

print("\nTraining started...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------------------
# 5. COMPARE TRAINING AND TEST PERFORMANCE
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING VS TEST PERFORMANCE")
print("=" * 60)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print(f"\nTraining accuracy: {train_accuracy:.4f} "
      f"({train_accuracy * 100:.2f}%)")

print(f"Test accuracy:     {test_accuracy:.4f} "
      f"({test_accuracy * 100:.2f}%)")

print(f"Accuracy gap:      "
      f"{(train_accuracy - test_accuracy) * 100:.2f} percentage points")

print("\nTraining classification report:")
print(
    classification_report(
        y_train,
        train_pred,
        labels=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

print("\nTest classification report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

# --------------------------------------------------
# 6. CONFUSION MATRIX ON TEST DATA
# --------------------------------------------------

print("\nTest confusion matrix:")
cm = confusion_matrix(
    y_test,
    test_pred,
    labels=["Normal", "Anomaly"],
)
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"],
)

disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest - Binary Classification")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "random_forest_binary_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# --------------------------------------------------
# 7. FEATURE IMPORTANCE
# --------------------------------------------------

print("\nTop 10 feature importances:")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(importance.head(10).to_string())

# --------------------------------------------------
# 8. SAVE MODEL AND RESULTS
# --------------------------------------------------

model_path = BASE_DIR / "random_forest_binary.joblib"

joblib.dump(model, model_path)

results = pd.DataFrame({
    "Metric": ["Training Accuracy", "Test Accuracy"],
    "Score": [train_accuracy, test_accuracy],
})

results.to_csv(
    BASE_DIR / "random_forest_binary_results.csv",
    index=False,
)

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)
print("Model: random_forest_binary.joblib")
print("Confusion matrix: random_forest_binary_confusion_matrix.png")
print("Accuracy results: random_forest_binary_results.csv")

print("\nEvaluation completed. No source datasets were modified.")
