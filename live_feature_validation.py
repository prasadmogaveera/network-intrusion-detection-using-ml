
import pandas as pd
import numpy as np
from pathlib import Path

file_path = Path("IoT Network Intrusion Dataset.csv")

if not file_path.exists():
    print("Dataset not found. Check the CSV filename and folder.")
    raise SystemExit

df = pd.read_csv(file_path, low_memory=False)

print("LIVE FEATURE VALIDATION")
print("=" * 45)

# 1. Packet counts
packet_counts = df["Tot_Fwd_Pkts"] + df["Tot_Bwd_Pkts"]
print("\n1. Total packet count")
print("Minimum:", packet_counts.min())
print("Median:", packet_counts.median())
print("Maximum:", packet_counts.max())

# 2. Total bytes
total_bytes = df["TotLen_Fwd_Pkts"] + df["TotLen_Bwd_Pkts"]
print("\n2. Total packet bytes")
print("Minimum:", total_bytes.min())
print("Median:", total_bytes.median())
print("Maximum:", total_bytes.max())

# 3. Packet length consistency
calculated_bytes = df["Pkt_Len_Mean"] * packet_counts
difference = (calculated_bytes - total_bytes).abs()

print("\n3. Packet length consistency")
print("Rows with non-zero difference:", (difference > 1).sum())
print("Median absolute difference:", difference.median())
print("95th percentile difference:", difference.quantile(0.95))

# 4. Flow duration
duration = df["Flow_Duration"]
print("\n4. Flow duration")
print("Minimum:", duration.min())
print("Median:", duration.median())
print("Maximum:", duration.max())
print("Zero-duration rows:", (duration == 0).sum())

# 5. Invalid or infinite values
numeric = df.select_dtypes(include="number")
print("\n5. Numeric data quality")
print("Infinite values:", np.isinf(numeric.to_numpy()).sum())
print("NaN values:", numeric.isna().sum().sum())

# 6. Protocol distribution
print("\n6. Protocol distribution")
print(df["Protocol"].value_counts(dropna=False).to_string())

print("\nValidation complete. No dataset changes made.")
