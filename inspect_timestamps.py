
import pandas as pd

file_path = "IoT Network Intrusion Dataset.csv"

print("Loading timestamp column...")

df = pd.read_csv(file_path, usecols=["Timestamp"])

print("\nFirst 10 timestamps:")
print(df["Timestamp"].head(10).to_string(index=False))

print("\nTimestamp data type:")
print(df["Timestamp"].dtype)

print("\nMissing timestamps:")
print(df["Timestamp"].isna().sum())

print("\nUnique timestamps:")
print(df["Timestamp"].nunique())

parsed = pd.to_datetime(df["Timestamp"], errors="coerce")

print("\nTimestamps that could not be parsed:")
print(parsed.isna().sum())

print("\nEarliest timestamp:")
print(parsed.min())

print("\nLatest timestamp:")
print(parsed.max())

print("\nAre timestamps already in chronological order?")
print(parsed.is_monotonic_increasing)
