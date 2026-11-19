
import pandas as pd

FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"

df = pd.read_csv(FILE)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

# Use unique timestamps, not individual rows.
times = pd.Series(df["Timestamp"].drop_duplicates().sort_values())
gaps = times.diff().dropna().dt.total_seconds()

print("Unique timestamps:", len(times))

print("\nTime gaps between consecutive timestamps (seconds):")
print("Minimum:", gaps.min())
print("Median:", gaps.median())
print("Mean:", round(gaps.mean(), 2))
print("Maximum:", gaps.max())

print("\nGap percentiles:")
print(gaps.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

print("\nGaps greater than 60 seconds:", int((gaps > 60).sum()))
print("Gaps greater than 5 minutes:", int((gaps > 300).sum()))

print("\nTimestamp range:")
print("Start:", times.min())
print("End:", times.max())
