
import pandas as pd

ORIGINAL_FILE = "IoT Network Intrusion Dataset.csv"
CLEAN_FILE = "IoT_Network_Intrusion_Clean_Binary.csv"
OUTPUT_FILE = "IoT_Network_Intrusion_LSTM_Clean.csv"

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

print("Loading original and cleaned datasets...")

original = pd.read_csv(ORIGINAL_FILE)
clean = pd.read_csv(CLEAN_FILE)

original["Timestamp"] = pd.to_datetime(
    original["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

# Keep only the information needed for sequence construction.
original = original[
    FEATURES + ["Timestamp", "Src_IP", "Dst_IP", "Label"]
].copy()

# Match the exact feature values and binary labels used in cleaning.
# The cleaned dataset contains one row per retained feature vector/label.
clean_keys = clean[FEATURES + ["Label"]].drop_duplicates()

# Keep only original records whose feature vector and label exist
# in the cleaned binary dataset.
matched = original.merge(
    clean_keys,
    on=FEATURES + ["Label"],
    how="inner"
)

# This step preserves timestamps and IP metadata.
# It does not claim to reconstruct which exact duplicate was retained
# by the original cleaning script.
matched = matched.dropna(subset=["Timestamp"])

matched = matched.sort_values("Timestamp").reset_index(drop=True)

matched.to_csv(OUTPUT_FILE, index=False)

print("\nLSTM cleaned data saved.")
print("Output file:", OUTPUT_FILE)
print("Rows:", len(matched))
print("Columns:", len(matched.columns))

print("\nLabel distribution:")
print(matched["Label"].value_counts())

print("\nDate and label distribution:")
matched["Date"] = matched["Timestamp"].dt.date
print(pd.crosstab(matched["Date"], matched["Label"]))

print("\nImportant:")
print("This is a timestamp-enriched mapping, not yet the final")
print("LSTM split or sequence dataset.")
