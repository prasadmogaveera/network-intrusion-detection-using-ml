
import pandas as pd
from pathlib import Path

# Original IoT dataset
file_path = Path("IoT Network Intrusion Dataset.csv")

if not file_path.exists():
    print("Dataset not found!")
    print("Check that the CSV is in your project folder.")
    raise SystemExit

df = pd.read_csv(file_path, low_memory=False)

print("DATASET FEATURE COMPATIBILITY CHECK")
print("-" * 45)

# Features we want to investigate for live traffic
candidate_features = [
    "Protocol",
    "Src_Port",
    "Dst_Port",
    "Flow_Duration",
    "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts",
    "Pkt_Len_Min",
    "Pkt_Len_Max",
    "Pkt_Len_Mean",
    "Pkt_Len_Std",
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "SYN_Flag_Cnt",
    "ACK_Flag_Cnt",
    "RST_Flag_Cnt",
    "FIN_Flag_Cnt",
    "PSH_Flag_Cnt",
]

for feature in candidate_features:
    if feature in df.columns:
        print(f"{feature}: AVAILABLE")
    else:
        print(f"{feature}: MISSING")

print("\nLabel columns:")
for col in ["Label", "Cat", "Sub_Cat"]:
    if col in df.columns:
        print(f"\n{col} distribution:")
        print(df[col].value_counts(dropna=False).to_string())

print("\nMissing values in candidate features:")
available = [c for c in candidate_features if c in df.columns]
print(df[available].isna().sum().to_string())

print("\nAudit complete. The CSV has not been modified.")
