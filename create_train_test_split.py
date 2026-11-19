
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

DATASETS = {
    "Binary": ("IoT_Network_Intrusion_Clean_Binary.csv", "Label"),
    "Multiclass": (
        "IoT_Network_Intrusion_Clean_Multiclass.csv", "Cat"
    )
}

for name, (filename, target) in DATASETS.items():
    print("\n" + "=" * 55)
    print(f"{name.upper()} TRAIN/TEST SPLIT")
    print("=" * 55)

    df = pd.read_csv(BASE_DIR / filename)

    # Assign the same group ID to identical feature vectors.
    groups = df.groupby(
        FEATURES, sort=False, dropna=False
    ).ngroup()

    # Create approximately an 80/20 split while preserving
    # class proportions as much as possible.
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(df[FEATURES], df[target], groups=groups)
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Verify that no feature vector crosses the split.
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)

    assert len(overlap) == 0, (
        "Leakage detected: feature groups overlap!"
    )

    train_file = f"{name.lower()}_train.csv"
    test_file = f"{name.lower()}_test.csv"

    train_df.to_csv(BASE_DIR / train_file, index=False)
    test_df.to_csv(BASE_DIR / test_file, index=False)

    print("Total rows:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training percentage:",
          round(len(train_df) / len(df) * 100, 2))
    print("Testing percentage:",
          round(len(test_df) / len(df) * 100, 2))
    print("Feature groups shared across splits:", len(overlap))

    print("\nTraining class distribution:")
    print(train_df[target].value_counts())

    print("\nTesting class distribution:")
    print(test_df[target].value_counts())

print("\nSplit complete. No models were trained.")
