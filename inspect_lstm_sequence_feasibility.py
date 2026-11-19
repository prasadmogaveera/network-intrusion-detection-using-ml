
import pandas as pd

FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"

df = pd.read_csv(FILE)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

print("Total records:", len(df))
print("Unique timestamps:", df["Timestamp"].nunique())
print("Unique source IPs:", df["Src_IP"].nunique())
print("Unique destination IPs:", df["Dst_IP"].nunique())

print("\nRecords per timestamp:")
counts = df.groupby("Timestamp").size()

print("Minimum:", counts.min())
print("Maximum:", counts.max())
print("Average:", round(counts.mean(), 2))
print("Timestamps with multiple records:", int((counts > 1).sum()))

print("\nLabels by date:")
df["Date"] = df["Timestamp"].dt.date
print(pd.crosstab(df["Date"], df["Label"]))

print("\nRecords in chronological order:", df["Timestamp"].is_monotonic_increasing)
