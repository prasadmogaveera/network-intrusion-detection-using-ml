
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

files = {
    "Binary": (
        "IoT_Network_Intrusion_Clean_Binary.csv",
        "Label",
    ),
    "Multiclass": (
        "IoT_Network_Intrusion_Clean_Multiclass.csv",
        "Cat",
    ),
}

features = [
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

for name, (filename, target) in files.items():
    print("\n" + "=" * 55)
    print(f"{name.upper()} DATASET VALIDATION")
    print("=" * 55)

    path = BASE_DIR / filename

    if not path.exists():
        print(f"ERROR: File not found: {filename}")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Expected columns:", len(features) + 1)
    print("Actual columns:", len(df.columns))

    missing_columns = [
        col for col in features + [target]
        if col not in df.columns
    ]
    extra_columns = [
        col for col in df.columns
        if col not in features + [target]
    ]

    print("Missing required columns:", missing_columns)
    print("Unexpected extra columns:", extra_columns)

    if missing_columns:
        print("Validation failed: required columns are missing.")
        continue

    X = df[features].apply(pd.to_numeric, errors="coerce")
    invalid_features = (
        X.isna().any(axis=1)
        | ~np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    )

    print("Invalid feature rows:", int(invalid_features.sum()))
    print("Missing target values:", int(df[target].isna().sum()))
    print("Exact duplicate rows:", int(df.duplicated().sum()))

    # Count feature vectors appearing more than once.
    repeated = X.duplicated(keep=False)
    print("Rows belonging to repeated feature vectors:",
          int(repeated.sum()))

    # Check whether identical feature vectors have different labels.
    check = pd.concat([X, df[target].rename(target)], axis=1)
    label_counts = check.groupby(features, dropna=False)[target].nunique()
    conflicting_vectors = int((label_counts > 1).sum())

    print("Conflicting feature vectors:", conflicting_vectors)
    print("\nClass distribution:")
    print(df[target].value_counts())

print("\nValidation complete. No model training was performed.")
