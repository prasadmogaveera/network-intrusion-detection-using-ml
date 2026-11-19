
import pandas as pd
import numpy as np

FILE_PATH = "IoT Network Intrusion Dataset.csv"

print("Loading dataset...")
df = pd.read_csv(FILE_PATH, low_memory=False)

# Reconstruct aggregate features
total_packets = df["Tot_Fwd_Pkts"] + df["Tot_Bwd_Pkts"]
total_bytes = df["TotLen_Fwd_Pkts"] + df["TotLen_Bwd_Pkts"]

print("\n" + "=" * 60)
print("1. PACKET COUNT CONSISTENCY")
print("=" * 60)
print("Rows with fewer than 1 packet:", (total_packets < 1).sum())
print("Rows with zero packets:", (total_packets == 0).sum())
print("Total packet count distribution:")
print(total_packets.describe().to_string())

print("\n" + "=" * 60)
print("2. PACKET LENGTH CONSISTENCY")
print("=" * 60)

# Check whether mean packet length roughly agrees with total bytes
estimated_bytes = df["Pkt_Len_Mean"] * total_packets
difference = (estimated_bytes - total_bytes).abs()

print("Rows where estimated and recorded bytes differ:",
      (difference > 1).sum())
print("Median absolute difference:", difference.median())
print("95th percentile absolute difference:",
      difference.quantile(0.95))

print("\nRows where packet-length statistics are inconsistent:")
bad_lengths = (
    (df["Pkt_Len_Min"] > df["Pkt_Len_Max"])
    | (df["Pkt_Len_Mean"] < df["Pkt_Len_Min"])
    | (df["Pkt_Len_Mean"] > df["Pkt_Len_Max"])
    | (df["Pkt_Len_Std"] < 0)
)
print("Count:", bad_lengths.sum())

print("\n" + "=" * 60)
print("3. INTER-ARRIVAL TIME CONSISTENCY")
print("=" * 60)

iat_bad = (
    (df["Flow_IAT_Min"] > df["Flow_IAT_Max"])
    | (df["Flow_IAT_Mean"] < df["Flow_IAT_Min"])
    | (df["Flow_IAT_Mean"] > df["Flow_IAT_Max"])
    | (df["Flow_IAT_Std"] < 0)
)

print("Rows with inconsistent IAT statistics:", iat_bad.sum())
print("Rows with zero Flow_Duration:",
      (df["Flow_Duration"] == 0).sum())

print("\n" + "=" * 60)
print("4. PROTOCOL 0 INVESTIGATION")
print("=" * 60)

protocol_zero = df[df["Protocol"] == 0]

print("Number of protocol 0 rows:", len(protocol_zero))
print("\nCategory counts:")
print(protocol_zero["Cat"].value_counts().to_string())

print("\nSubcategory counts:")
print(protocol_zero["Sub_Cat"].value_counts().to_string())

print("\nSample records:")
print(
    protocol_zero[
        [
            "Protocol", "Src_Port", "Dst_Port",
            "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
            "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
            "Pkt_Len_Min", "Pkt_Len_Max",
            "Label", "Cat", "Sub_Cat"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("5. DIRECTION-INDEPENDENT PORT CHECK")
print("=" * 60)

print("Rows where source and destination ports are equal:",
      (df["Src_Port"] == df["Dst_Port"]).sum())

print("\nSample rows with port 0:")
print(
    df.loc[
        (df["Src_Port"] == 0) | (df["Dst_Port"] == 0),
        ["Protocol", "Src_Port", "Dst_Port", "Cat", "Sub_Cat"]
    ].head(10).to_string(index=False)
)

print("\nValidation complete. No data was modified.")
