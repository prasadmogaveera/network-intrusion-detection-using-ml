
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.metrics import precision_recall_fscore_support
import joblib


# -----------------------------
# 1. Configuration
# -----------------------------
SEED = 42
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.20

DATA_FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# -----------------------------
# 2. Load and validate data
# -----------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} was not found. "
        "Place this script in the folder containing the CSV."
    )

df = pd.read_csv(DATA_FILE)

required_columns = FEATURES + ["Label", "Timestamp"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce", dayfirst=True
)

df = df.dropna(subset=FEATURES + ["Label", "Timestamp"]).copy()
df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES)

df["Target"] = df["Label"].map({
    "Normal": 0,
    "Anomaly": 1
})

df = df.dropna(subset=["Target"]).copy()
df["Target"] = df["Target"].astype(int)
df["Date"] = df["Timestamp"].dt.date

print("Dataset shape:", df.shape)
print("\nClass counts:")
print(df["Label"].value_counts())


# -----------------------------
# 3. Build non-overlapping sequences
# -----------------------------
# Sort by time and never combine different dates.
# Non-overlapping windows reduce leakage from shared time steps.
df = df.sort_values("Timestamp").reset_index(drop=True)

X_sequences = []
y_sequences = []

for date, day_df in df.groupby("Date", sort=True):
    day_df = day_df.sort_values("Timestamp").reset_index(drop=True)

    values = day_df[FEATURES].to_numpy(dtype=np.float32)
    labels = day_df["Target"].to_numpy(dtype=np.int32)

    for start in range(0, len(day_df) - SEQUENCE_LENGTH + 1,
                       SEQUENCE_LENGTH):

        end = start + SEQUENCE_LENGTH

        # Only use windows whose labels are consistent.
        window_labels = labels[start:end]

        if not np.all(window_labels == window_labels[-1]):
            continue

        X_sequences.append(values[start:end])
        y_sequences.append(window_labels[-1])

X = np.asarray(X_sequences, dtype=np.float32)
y = np.asarray(y_sequences, dtype=np.int32)

if len(X) == 0 or len(np.unique(y)) != 2:
    raise ValueError(
        "Could not create sequences containing both classes. "
        "Check the dataset and sequence-building step."
    )

print("\nSequence shape:", X.shape)
print("Target shape:", y.shape)
print("Sequence length:", SEQUENCE_LENGTH)
print("Normal sequences:", int(np.sum(y == 0)))
print("Anomaly sequences:", int(np.sum(y == 1)))


# -----------------------------
# 4. Split sequences into train/test
# -----------------------------
# This is a demonstration split, not an independent
# unseen-session evaluation.
indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y
)

X_train = X[train_idx]
X_test = X[test_idx]
y_train = y[train_idx]
y_test = y[test_idx]

print("\nTraining sequences:", len(X_train))
print("Testing sequences:", len(X_test))


# -----------------------------
# 5. Scale features without test leakage
# -----------------------------
scaler = StandardScaler()

# Fit only on time steps belonging to training sequences.
n_features = X_train.shape[2]

X_train_scaled = scaler.fit_transform(
    X_train.reshape(-1, n_features)
).reshape(X_train.shape)

X_test_scaled = scaler.transform(
    X_test.reshape(-1, n_features)
).reshape(X_test.shape)

joblib.dump(scaler, "lstm_binary_scaler.joblib")


# -----------------------------
# 6. Handle class imbalance
# -----------------------------
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# -----------------------------
# 7. Build the LSTM model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(SEQUENCE_LENGTH, n_features)
    ),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# -----------------------------
# 8. Train the model
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.15,
    epochs=15,
    batch_size=256,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
    shuffle=True
)


# -----------------------------
# 9. Evaluate on test sequences
# -----------------------------
probabilities = model.predict(
    X_test_scaled, batch_size=256, verbose=0
).ravel()

y_pred = (probabilities >= 0.5).astype(int)

print("\nTEST CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Anomaly"],
    digits=4,
    zero_division=0
))

accuracy = accuracy_score(y_test, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Anomaly Precision: {precision:.4f}")
print(f"Anomaly Recall:    {recall:.4f}")
print(f"Anomaly F1-score:  {f1:.4f}")


# -----------------------------
# 10. Save results and model
# -----------------------------
model.save("lstm_binary_demo.keras")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Anomaly"]
)

disp.plot(values_format="d")
plt.title("LSTM Binary Classification - Test Set")
plt.tight_layout()
plt.savefig("lstm_binary_demo_confusion_matrix.png", dpi=200)
plt.close()

pd.DataFrame([{
    "Model": "LSTM",
    "Sequence_Length": SEQUENCE_LENGTH,
    "Train_Sequences": len(X_train),
    "Test_Sequences": len(X_test),
    "Accuracy": accuracy,
    "Anomaly_Precision": precision,
    "Anomaly_Recall": recall,
    "Anomaly_F1": f1
}]).to_csv("lstm_binary_demo_results.csv", index=False)

plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training History")
plt.legend()
plt.tight_layout()
plt.savefig("lstm_binary_demo_training_history.png", dpi=200)
plt.close()

print("\nTraining completed.")
print("Saved model: lstm_binary_demo.keras")
print("Saved scaler: lstm_binary_scaler.joblib")
print("Saved metrics: lstm_binary_demo_results.csv")
print("Saved confusion matrix: lstm_binary_demo_confusion_matrix.png")
print("Saved training graph: lstm_binary_demo_training_history.png")
