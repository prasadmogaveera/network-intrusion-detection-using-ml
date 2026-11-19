
import pandas as pd

INPUT_FILE = "IoT Network Intrusion Dataset.csv"
OUTPUT_FILE = "IoT_Network_Intrusion_LSTM_Sequence_Data.csv"

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

print("Loading original dataset...")

df = pd.read_csv(INPUT_FILE)

required = FEATURES + [
    "Timestamp", "Src_IP", "Dst_IP", "Label"
]

df = df[required].copy()

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

df = df.dropna(subset=["Timestamp", "Label"])

# Remove exact duplicate records across selected features and target.
# Keep the first associated timestamp and IP metadata.
df = df.drop_duplicates(
    subset=FEATURES + ["Label"],
    keep="first"
)

df = df.sort_values("Timestamp").reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nLSTM sequence-ready data saved.")
print("Output:", OUTPUT_FILE)
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nLabel distribution:")
print(df["Label"].value_counts())
print("\nFirst 5 chronological records:")
print(df[["Timestamp", "Src_IP", "Dst_IP", "Label"]].head())
