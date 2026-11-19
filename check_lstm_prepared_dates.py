
import pandas as pd

df = pd.read_csv(
    "IoT_Network_Intrusion_LSTM_Sequence_Data.csv",
    usecols=["Timestamp", "Label"]
)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Date"] = df["Timestamp"].dt.date

print("\nRows and labels per date:")
print(
    pd.crosstab(df["Date"], df["Label"]).to_string()
)

print("\nTotal rows per date:")
print(df.groupby("Date").size().to_string())
