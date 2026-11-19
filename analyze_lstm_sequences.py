
import pandas as pd

file_path = "IoT Network Intrusion Dataset.csv"

df = pd.read_csv(
    file_path,
    usecols=["Timestamp", "Label", "Cat"]
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

print("\n1. Rows per timestamp")
counts = df.groupby("Timestamp").size()

print("Minimum rows:", counts.min())
print("Maximum rows:", counts.max())
print("Average rows:", round(counts.mean(), 2))
print("Timestamps with multiple rows:", (counts > 1).sum())

print("\n2. Rows per date")
daily = df.groupby(df["Timestamp"].dt.date).size()

print(daily.describe())

print("\n3. Labels per timestamp")
labels_per_time = df.groupby("Timestamp")["Label"].nunique()

print(
    "Timestamps containing both Normal and Anomaly:",
    (labels_per_time > 1).sum()
)

print("\n4. Category distribution")
print(df["Cat"].value_counts())

print("\n5. First 10 chronological rows")
print(
    df.sort_values("Timestamp")[
        ["Timestamp", "Label", "Cat"]
    ].head(10).to_string(index=False)
)
