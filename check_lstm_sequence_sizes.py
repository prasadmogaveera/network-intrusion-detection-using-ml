
import pandas as pd

FILE = "IoT_Network_Intrusion_LSTM_Clean_v2.csv"

df = pd.read_csv(FILE)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

# Count unique timestamps within each date.
df["Date"] = df["Timestamp"].dt.date

print("Unique timestamps per date:\n")

for date, group in df.groupby("Date"):
    timestamps = group["Timestamp"].nunique()
    label = group["Label"].iloc[0]

    # Number of full sequences possible at sequence length 10,
    # assuming one sample per unique timestamp.
    possible_sequences = max(0, timestamps - 10 + 1)

    print(
        f"{date} | Label: {label} | "
        f"Timestamps: {timestamps} | "
        f"Possible length-10 windows: {possible_sequences}"
    )
