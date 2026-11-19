
import pandas as pd

FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"

df = pd.read_csv(FILE)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

# Count distinct labels at each timestamp.
labels_per_time = df.groupby("Timestamp")["Label"].nunique()

print("Total timestamps:", len(labels_per_time))
print(
    "Timestamps containing multiple labels:",
    int((labels_per_time > 1).sum())
)

# Inspect how labels are distributed across dates.
df["Date"] = df["Timestamp"].dt.date

print("\nNumber of dates:", df["Date"].nunique())
print("\nLabels per date:")
print(df.groupby("Date")["Label"].nunique())

print("\nFirst 10 chronological records:")
print(df[["Timestamp", "Protocol", "Src_Port", "Dst_Port", "Label"]].head(10))

print("\nLast 10 chronological records:")
print(df[["Timestamp", "Protocol", "Src_Port", "Dst_Port", "Label"]].tail(10))
