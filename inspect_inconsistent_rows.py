
import pandas as pd
import numpy as np

df = pd.read_csv("IoT Network Intrusion Dataset.csv")

columns = [
    "Protocol",
    "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts",
    "Pkt_Len_Mean",
    "Pkt_Len_Std",
    "Pkt_Len_Min",
    "Pkt_Len_Max",
    "Flow_Duration",
    "Flow_Byts/s",
    "Flow_Pkts/s",
    "Label",
    "Cat",
    "Sub_Cat"
]

# Inspect rows where a packet count is zero but its byte total is positive.
forward_issue = (
    (df["Tot_Fwd_Pkts"] == 0) &
    (df["TotLen_Fwd_Pkts"] > 0)
)

backward_issue = (
    (df["Tot_Bwd_Pkts"] == 0) &
    (df["TotLen_Bwd_Pkts"] > 0)
)

print("\n1. ZERO PACKET COUNT WITH POSITIVE BYTES")
print("=" * 55)
print("Forward:", int(forward_issue.sum()))
print("Backward:", int(backward_issue.sum()))

print("\nExamples:")
print(
    df.loc[forward_issue | backward_issue, columns]
      .head(10)
      .to_string(index=False)
)

print("\n2. INFINITE RATE VALUES")
print("=" * 55)

for col in ["Flow_Byts/s", "Flow_Pkts/s"]:
    mask = np.isinf(df[col])
    print(f"\n{col}: {int(mask.sum())} infinite values")
    print(
        df.loc[mask, [
            "Flow_Duration",
            "Tot_Fwd_Pkts",
            "Tot_Bwd_Pkts",
            "TotLen_Fwd_Pkts",
            "TotLen_Bwd_Pkts",
            col,
            "Cat",
            "Sub_Cat"
        ]].head(5).to_string(index=False)
    )

print("\n3. CHECK COMPLETE")
print("The original dataset was not modified.")
