
import pandas as pd
import numpy as np

CSV_FILE = "IoT Network Intrusion Dataset.csv"

df = pd.read_csv(CSV_FILE)

# Compare recorded bytes with mean packet length × packet count.
df["Total_Pkts"] = df["Tot_Fwd_Pkts"] + df["Tot_Bwd_Pkts"]

df["Total_Bytes"] = (
    df["TotLen_Fwd_Pkts"] + df["TotLen_Bwd_Pkts"]
)

df["Estimated_Bytes"] = df["Pkt_Len_Mean"] * df["Total_Pkts"]

df["Byte_Difference"] = (
    df["Total_Bytes"] - df["Estimated_Bytes"]
).abs()

print("\n1. MISMATCHES BY PROTOCOL AND ATTACK CATEGORY")
print("=" * 55)

df["Mismatch"] = df["Byte_Difference"] > 1

summary = (
    df.groupby(["Protocol", "Cat"], dropna=False)
    .agg(
        Total_Rows=("Mismatch", "size"),
        Mismatch_Rows=("Mismatch", "sum")
    )
    .reset_index()
)

summary["Mismatch_Percent"] = (
    100 * summary["Mismatch_Rows"] / summary["Total_Rows"]
).round(2)

print(summary.to_string(index=False))

print("\n2. EXAMPLE MISMATCHED TCP ROWS")
print("=" * 55)

columns = [
    "Protocol", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Mean", "Byte_Difference", "Cat"
]

tcp_examples = df[
    (df["Protocol"] == 6) & df["Mismatch"]
]

if len(tcp_examples):
    print(tcp_examples[columns].head(5).to_string(index=False))
else:
    print("No TCP mismatches found.")

print("\n3. INFINITE VALUES BY ATTACK CATEGORY")
print("=" * 55)

for col in ["Flow_Byts/s", "Flow_Pkts/s"]:
    print(f"\n{col}")
    mask = np.isinf(df[col])
    print(df.loc[mask, "Cat"].value_counts(dropna=False).to_string())

print("\n4. CHECK COMPLETE")
print("Original dataset was not modified.")
