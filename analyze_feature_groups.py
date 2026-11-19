
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

features = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

datasets = {
    "Binary": ("IoT_Network_Intrusion_Clean_Binary.csv", "Label"),
    "Multiclass": (
        "IoT_Network_Intrusion_Clean_Multiclass.csv", "Cat"
    )
}

for name, (filename, target) in datasets.items():
    print("\n" + "=" * 55)
    print(name.upper(), "FEATURE GROUP ANALYSIS")
    print("=" * 55)

    df = pd.read_csv(BASE_DIR / filename)

    # Count distinct feature vectors overall.
    unique_features = df[features].drop_duplicates()
    print("Total rows:", len(df))
    print("Unique feature vectors:", len(unique_features))
    print("Rows beyond unique feature vectors:",
          len(df) - len(unique_features))

    # Count unique feature vectors for each class.
    per_class = df.groupby(target).apply(
        lambda part: part[features].drop_duplicates().shape[0],
        include_groups=False
    )

    print("\nUnique feature vectors per class:")
    print(per_class)

    # Check whether a feature vector belongs to multiple classes.
    label_counts = df.groupby(features, dropna=False)[target].nunique()
    print("\nFeature vectors shared across different classes:",
          int((label_counts > 1).sum()))

print("\nAnalysis complete. No files were modified.")
