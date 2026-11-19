
import pandas as pd

file_path = "IoT Network Intrusion Dataset.csv"

df = pd.read_csv(
    file_path,
    usecols=["Timestamp", "Src_IP", "Dst_IP", "Label", "Cat"]
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

# Sort by time for inspection only
df = df.sort_values("Timestamp").reset_index(drop=True)

print("\n1. Unique labels per date")
df["Date"] = df["Timestamp"].dt.date

print(
    df.groupby("Date")["Label"].nunique().to_string()
)

print("\n2. Number of source IP addresses")
print(df["Src_IP"].nunique())

print("\n3. Number of destination IP addresses")
print(df["Dst_IP"].nunique())

print("\n4. Most common source IP addresses")
print(df["Src_IP"].value_counts().head(10))

print("\n5. Most common source-destination pairs")
pairs = df.groupby(["Src_IP", "Dst_IP"]).size()
print(pairs.sort_values(ascending=False).head(10))

print("\n6. Label distribution by date")
print(pd.crosstab(df["Date"], df["Label"]).to_string())
