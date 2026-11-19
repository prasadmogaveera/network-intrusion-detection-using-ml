from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "binary_train.csv"
TEST_PATH = BASE_DIR / "binary_test.csv"
LIVE_PATH = BASE_DIR / "live_features.csv"
MODEL_PATH = BASE_DIR / "random_forest_live_adapted.joblib"
REPORT_PATH = BASE_DIR / "live_adaptation_report.txt"

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts", "Pkt_Len_Min", "Pkt_Len_Max",
    "Pkt_Len_Mean", "Pkt_Len_Std", "Flow_IAT_Mean",
    "Flow_IAT_Std", "SYN_Flag_Cnt", "ACK_Flag_Cnt",
    "RST_Flag_Cnt", "FIN_Flag_Cnt", "PSH_Flag_Cnt",
]
FLAGS = [
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt",
]


def load_features(path):
    frame = pd.read_csv(path)
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing features: {missing}")
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{path.name} contains invalid feature values.")
    values = values.copy()
    for column in FLAGS:
        values[column] = (values[column] > 0).astype(int)
    return values


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    live = load_features(LIVE_PATH)

    if len(live) < 20:
        raise ValueError("At least 20 captured live flows are required.")

    x_train = load_features(TRAIN_PATH)
    x_test = load_features(TEST_PATH)
    y_train = train["Label"].astype(str)
    y_test = test["Label"].astype(str)

    # Captured browsing traffic is treated as provisional Normal data. Its
    # larger sample weight gives the new model a measurable live-domain signal
    # without modifying the original training files or model.
    x_combined = pd.concat([x_train, live], ignore_index=True)
    y_combined = pd.concat(
        [y_train, pd.Series(["Normal"] * len(live))],
        ignore_index=True,
    )
    weights = np.ones(len(x_combined))
    weights[len(x_train):] = 1000.0

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_combined, y_combined, sample_weight=weights)
    joblib.dump(model, MODEL_PATH)

    test_pred = model.predict(x_test)
    live_pred = model.predict(live)
    test_proba = model.predict_proba(x_test)
    live_proba = model.predict_proba(live)
    anomaly_index = list(model.classes_).index("Anomaly")

    report = [
        "LIVE-ADAPTED RANDOM FOREST REPORT",
        "=" * 50,
        f"Live provisional Normal rows: {len(live)}",
        f"Live sample weight: 1000",
        f"Original held-out test accuracy: {accuracy_score(y_test, test_pred):.6f}",
        f"Live Anomaly predictions: {(live_pred == 'Anomaly').sum()}",
        f"Live Normal predictions: {(live_pred == 'Normal').sum()}",
        f"Live mean P(Anomaly): {live_proba[:, anomaly_index].mean():.6f}",
        "",
        "Held-out test classification report:",
        classification_report(
            y_test, test_pred, labels=["Normal", "Anomaly"],
            digits=4, zero_division=0,
        ),
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
