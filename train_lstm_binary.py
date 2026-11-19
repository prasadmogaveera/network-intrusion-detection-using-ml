
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# -------------------------------
# 1. Configuration
# -------------------------------
DATA_FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"
SEQUENCE_LENGTH = 10
RANDOM_STATE = 42

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -------------------------------
# 2. Load and validate data
# -------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], errors="coerce"
)
df = df.dropna(subset=["Timestamp", "Label"])
df = df.sort_values("Timestamp").reset_index(drop=True)

required = FEATURES + ["Timestamp", "Label"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[FEATURES] = df[FEATURES].replace(
    [np.inf, -np.inf], np.nan
)
df = df.dropna(subset=FEATURES).copy()

# Keep this experiment binary.
df["Target"] = df["Label"].map({
    "Normal": 0,
    "Anomaly": 1
})

df = df.dropna(subset=["Target"]).copy()
df["Target"] = df["Target"].astype(int)
df["Date"] = df["Timestamp"].dt.date

print("Rows:", len(df))
print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nDate/label distribution:")
print(pd.crosstab(df["Date"], df["Label"]))

# -------------------------------
# 3. Create date-contained sequences
# -------------------------------
print("\nCreating sequences...")

X_sequences = []
y_sequences = []
sequence_dates = []

for date, group in df.groupby("Date", sort=True):
    group = group.sort_values("Timestamp")

    # One record per unique feature-label combination was retained
    # during preparation. This is an exploratory sequence dataset.
    X = group[FEATURES].to_numpy(dtype=np.float32)
    y = group["Target"].to_numpy(dtype=np.int32)

    for start in range(len(group) - SEQUENCE_LENGTH + 1):
        end = start + SEQUENCE_LENGTH

        X_sequences.append(X[start:end])
        y_sequences.append(y[end - 1])
        sequence_dates.append(date)

X_sequences = np.asarray(X_sequences, dtype=np.float32)
y_sequences = np.asarray(y_sequences, dtype=np.int32)
sequence_dates = np.asarray(sequence_dates)

if len(X_sequences) == 0:
    raise ValueError("No sequences could be created.")

print("Sequence shape:", X_sequences.shape)
print("Target shape:", y_sequences.shape)
print("Sequences never cross date boundaries.")

# -------------------------------
# 4. Stop before misleading evaluation
# -------------------------------
print("\nIMPORTANT EVALUATION LIMITATION")
print("Normal and Anomaly occur on separate dates.")
print("A date-held-out test cannot represent both classes.")
print("A random sequence split may leak scenario-specific patterns.")
print("Therefore, this script will NOT report a test score.")
print("We need to choose and justify a valid evaluation strategy")
print("before fitting the scaler or training the LSTM.")
