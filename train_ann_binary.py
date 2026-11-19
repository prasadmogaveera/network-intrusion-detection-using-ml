
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

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

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("ANN - BINARY CLASSIFICATION")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading datasets...")

train_df = pd.read_csv(BASE_DIR / "binary_train.csv")
test_df = pd.read_csv(BASE_DIR / "binary_test.csv")

X_train = train_df[FEATURES].astype(np.float32)
y_train = (train_df[TARGET] == "Anomaly").astype(np.int32)

X_test = test_df[FEATURES].astype(np.float32)
y_test = (test_df[TARGET] == "Anomaly").astype(np.int32)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", len(FEATURES))
print("Target encoding: Normal = 0, Anomaly = 1")

# --------------------------------------------------
# 3. SCALE FEATURES
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

# Fit only on training data.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed.")

# --------------------------------------------------
# 4. HANDLE CLASS IMBALANCE
# --------------------------------------------------

classes = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train,
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass weights:")
print("Normal:", round(class_weights[0], 4))
print("Anomaly:", round(class_weights[1], 4))

# --------------------------------------------------
# 5. BUILD ANN MODEL
# --------------------------------------------------

print("\nBuilding ANN model...")

model = Sequential([
    Input(shape=(len(FEATURES),)),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

print("\nTraining started...")

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=20,
    batch_size=1024,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1,
)

print("\nTraining completed!")

# --------------------------------------------------
# 7. PREDICT TRAINING AND TEST DATA
# --------------------------------------------------

print("\nEvaluating model...")

train_probabilities = model.predict(
    X_train_scaled, batch_size=4096, verbose=0
).ravel()

test_probabilities = model.predict(
    X_test_scaled, batch_size=4096, verbose=0
).ravel()

# Default decision threshold: 0.5.
train_pred = (train_probabilities >= 0.5).astype(int)
test_pred = (test_probabilities >= 0.5).astype(int)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\n" + "=" * 60)
print("TRAINING VS TEST PERFORMANCE")
print("=" * 60)

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
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

print("\nTest classification report:")
print(
    classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        digits=4,
        zero_division=0,
    )
)

# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1],
)

print("\nTest confusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"],
)

disp.plot(cmap="Blues", values_format="d")
plt.title("ANN - Binary Classification")
plt.tight_layout()

plt.savefig(
    BASE_DIR / "ann_binary_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# --------------------------------------------------
# 9. PLOT TRAINING HISTORY
# --------------------------------------------------

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("ANN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("ANN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(
    BASE_DIR / "ann_binary_training_history.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()

# --------------------------------------------------
# 10. SAVE MODEL, SCALER AND RESULTS
# --------------------------------------------------

model.save(BASE_DIR / "ann_binary.keras")
joblib.dump(scaler, BASE_DIR / "ann_binary_scaler.joblib")

pd.DataFrame({
    "Metric": ["Training Accuracy", "Test Accuracy"],
    "Score": [train_accuracy, test_accuracy],
}).to_csv(
    BASE_DIR / "ann_binary_results.csv",
    index=False,
)

print("\n" + "=" * 60)
print("SAVED OUTPUTS")
print("=" * 60)
print("Model: ann_binary.keras")
print("Scaler: ann_binary_scaler.joblib")
print("Confusion matrix: ann_binary_confusion_matrix.png")
print("Training history: ann_binary_training_history.png")
print("Results: ann_binary_results.csv")

print("\nANN evaluation completed.")
