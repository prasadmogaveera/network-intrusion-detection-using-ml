
import pandas as pd

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

for task in ["binary", "multiclass"]:
    print(f"\n--- {task.upper()} ---")

    train = pd.read_csv(f"{task}_train.csv")
    test = pd.read_csv(f"{task}_test.csv")

    # Compare feature vectors, excluding the target label.
    train_features = train[FEATURES].drop_duplicates()
    test_features = test[FEATURES].drop_duplicates()

    overlap = train_features.merge(
        test_features,
        on=FEATURES,
        how="inner"
    )

    print("Unique train feature vectors:", len(train_features))
    print("Unique test feature vectors:", len(test_features))
    print("Feature vectors shared across splits:", len(overlap))

print("\nCheck completed.")
