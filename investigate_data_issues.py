
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path("IoT Network Intrusion Dataset.csv")

if not file_path.exists():
    print("Dataset not found. Check the filename.")
    raise SystemExit

df = pd.read_csv(file_path, low_memory=False)

print("=" * 55)
print("INVESTIGATING DATA QUALITY ISSUES")
print("=" * 55)

# 1. Find exactly which columns contain infinite values
print("\n1. Infinite values by column")
numeric = df.select_dtypes(include="number")
infinite_counts = pd.Series(
    np.isinf(numeric.to_numpy()).sum(axis=0),
    index=numeric.columns
)
infinite_counts = infinite_counts[infinite_counts > 0]
print(infinite_counts.to_string() if not infinite_counts.empty
      else "No infinite values found.")

# 2. Check packet-length consistency by protocol
print("\n2. Packet-length consistency by protocol")
total_packets = df["Tot_Fwd_Pkts"] + df["Tot_Bwd_Pkts"]
total_bytes = df["TotLen_Fwd_Pkts"] + df["TotLen_Bwd_Pkts"]
estimated_bytes = df["Pkt_Len_Mean"] * total_packets
difference = (estimated_bytes - total_bytes).abs()

for protocol in sorted(df["Protocol"].dropna().unique()):
    mask = df["Protocol"] == protocol
    print(
        f"Protocol {protocol}: "
        f"rows={mask.sum()}, "
        f"differences > 1 byte={(difference[mask] > 1).sum()}"
    )

# 3. Inspect protocol 0 records
print("\n3. Protocol 0 label distribution")
protocol_zero = df[df["Protocol"] == 0]
print(protocol_zero["Cat"].value_counts().to_string())

print("\nProtocol 0 packet and byte statistics:")
columns = [
    "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts",
    "Flow_Duration",
]
print(protocol_zero[columns].describe().round(2).to_string())

# 4. Check the label balance
print("\n4. Binary label distribution")
print(df["Label"].value_counts().to_string())

print("\nInvestigation complete. Original CSV unchanged.")
