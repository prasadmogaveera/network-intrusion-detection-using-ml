
import pandas as pd

df = pd.read_csv(
    "IoT Network Intrusion Dataset.csv",
    usecols=["Timestamp", "Cat", "Sub_Cat"]
)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="%d/%m/%Y %I:%M:%S %p",
    errors="coerce"
)

df["Date"] = df["Timestamp"].dt.date

print("\n1. Attack categories per date:")
print(
    pd.crosstab(df["Date"], df["Cat"]).to_string()
)

print("\n2. Attack subcategories per date:")
print(
    pd.crosstab(df["Date"], df["Sub_Cat"]).to_string()
)

print("\n3. Number of unique timestamps per date:")
print(
    df.groupby("Date")["Timestamp"].nunique().to_string()
)

print("\n4. First and last timestamp per date:")
print(
    df.groupby("Date")["Timestamp"]
      .agg(["min", "max"])
      .to_string()
)
