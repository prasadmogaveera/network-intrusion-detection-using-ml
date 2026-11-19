
import pandas as pd
import numpy as np

FILE_PATH = "IoT Network Intrusion Dataset.csv"

print("Loading dataset... Please wait.")
df = pd.read_csv(FILE_PATH, low_memory=False)

print("\nDataset shape:", df.shape)

# Features that we may be able to reproduce using Scapy
required_columns = [
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
    "SYN_Flag_Cnt",
    "ACK_Flag_Cnt",
    "RST_Flag_Cnt",
    "FIN_Flag_Cnt",
    "PSH_Flag_Cnt",
    "Flow_IAT_Mean",
    "Flow_IAT_Std",
    "Flow_IAT_Max",
    "Flow_IAT_Min",
    "Cat",
    "Sub_Cat",
]

print("\n" + "=" * 60)
print("1. REQUIRED COLUMN CHECK")
print("=" * 60)

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
else:
    print("All candidate source columns are present.")

# Continue only if the required source columns exist
if not missing_columns:
    # Build candidate features without modifying the original dataset
    candidate = pd.DataFrame(index=df.index)

    candidate["Protocol"] = df["Protocol"]
    candidate["Min_Port"] = df[["Src_Port", "Dst_Port"]].min(axis=1)
    candidate["Max_Port"] = df[["Src_Port", "Dst_Port"]].max(axis=1)

    candidate["Flow_Duration"] = df["Flow_Duration"]

    candidate["Total_Packets"] = (
        df["Tot_Fwd_Pkts"] + df["Tot_Bwd_Pkts"]
    )

    candidate["Total_Bytes"] = (
        df["TotLen_Fwd_Pkts"] + df["TotLen_Bwd_Pkts"]
    )

    for col in [
        "Pkt_Len_Min",
        "Pkt_Len_Max",
        "Pkt_Len_Mean",
        "Pkt_Len_Std",
        "SYN_Flag_Cnt",
        "ACK_Flag_Cnt",
        "RST_Flag_Cnt",
        "FIN_Flag_Cnt",
        "PSH_Flag_Cnt",
        "Flow_IAT_Mean",
        "Flow_IAT_Std",
        "Flow_IAT_Max",
        "Flow_IAT_Min",
    ]:
        candidate[col] = df[col]

    print("\n" + "=" * 60)
    print("2. CANDIDATE FEATURE SUMMARY")
    print("=" * 60)
    print(candidate.describe().T.to_string())

    print("\n" + "=" * 60)
    print("3. PROTOCOL VALUES")
    print("=" * 60)
    print(df["Protocol"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 60)
    print("4. PORT CHECK")
    print("=" * 60)
    print("Source port missing values:", df["Src_Port"].isna().sum())
    print("Destination port missing values:", df["Dst_Port"].isna().sum())
    print("Source port range:", df["Src_Port"].min(), "to", df["Src_Port"].max())
    print("Destination port range:", df["Dst_Port"].min(), "to", df["Dst_Port"].max())

    print("\n" + "=" * 60)
    print("5. DURATION CHECK")
    print("=" * 60)
    duration = pd.to_numeric(df["Flow_Duration"], errors="coerce")
    print("Zero-duration flows:", (duration == 0).sum())
    print("Negative-duration flows:", (duration < 0).sum())
    print("Missing durations:", duration.isna().sum())
    print("Duration percentiles:")
    print(duration.quantile([0, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).to_string())

    print("\n" + "=" * 60)
    print("6. INVALID NUMERIC VALUES")
    print("=" * 60)
    numeric = candidate.select_dtypes(include=[np.number])

    for col in numeric.columns:
        values = numeric[col].to_numpy(dtype=float)
        print(
            f"{col}: missing={np.isnan(values).sum()}, "
            f"infinite={np.isinf(values).sum()}"
        )

    print("\n" + "=" * 60)
    print("7. CLASS DISTRIBUTION")
    print("=" * 60)
    print("\nCategory counts:")
    print(df["Cat"].value_counts(dropna=False).to_string())

    print("\nSubcategory counts:")
    print(df["Sub_Cat"].value_counts(dropna=False).to_string())

    print("\n" + "=" * 60)
    print("8. FINAL CANDIDATE FEATURE LIST")
    print("=" * 60)
    print(candidate.columns.tolist())
    print("\nValidation finished. No files or dataset rows were changed.")
