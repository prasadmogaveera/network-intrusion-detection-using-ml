
import pandas as pd

CLEAN_FILE = "IoT_Network_Intrusion_Clean_Binary.csv"
LSTM_FILE = "IoT_Network_Intrusion_LSTM_Clean.csv"
OUTPUT_FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

print("Loading datasets...")

clean = pd.read_csv(CLEAN_FILE)
lstm = pd.read_csv(LSTM_FILE)

# Build keys for feature vectors retained by the cleaner.
clean_keys = clean[FEATURES + ["Label"]].drop_duplicates()

# Keep one metadata record for each feature vector and label.
# This prevents repeated original records from multiplying rows.
lstm_unique = lstm.drop_duplicates(
    subset=FEATURES + ["Label"],
    keep="first"
).copy()

# Keep only feature vectors that exist in the cleaned dataset.
corrected = lstm_unique.merge(
    clean_keys,
    on=FEATURES + ["Label"],
    how="inner",
    validate="one_to_one"
)

corrected["Timestamp"] = pd.to_datetime(
    corrected["Timestamp"], errors="coerce"
)

corrected = corrected.dropna(subset=["Timestamp"])
corrected = corrected.sort_values("Timestamp").reset_index(drop=True)

corrected.to_csv(OUTPUT_FILE, index=False)

print("\nCorrected dataset saved:", OUTPUT_FILE)
print("Rows:", len(corrected))

print("\nLabel distribution:")
print(corrected["Label"].value_counts())

print("\nDate and label distribution:")
corrected["Date"] = corrected["Timestamp"].dt.date
print(pd.crosstab(corrected["Date"], corrected["Label"]))

print("\nNote: this keeps one timestamp per feature-label combination.")
print("It does not establish independent temporal evaluation.")
