from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_binary.joblib"
ADAPTED_MODEL_PATH = BASE_DIR / "random_forest_live_adapted.joblib"
TRAIN_PATH = BASE_DIR / "binary_train.csv"
TEST_PATH = BASE_DIR / "binary_test.csv"
FEATURES = [
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


def describe_features(frame, name):
    print(f"\n{name} feature statistics")
    print("-" * 80)
    print(
        frame[FEATURES]
        .describe(percentiles=[0.01, 0.50, 0.99])
        .T[["min", "1%", "50%", "99%", "max", "mean"]]
        .round(3)
        .to_string()
    )


def validate(frame, name):
    values = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    invalid = values.isna().any(axis=1) | ~np.isfinite(values).all(axis=1)
    print(f"{name} invalid rows: {int(invalid.sum())}/{len(frame)}")
    return values, invalid


def main():
    parser = argparse.ArgumentParser(
        description="Compare saved live flow features with training data."
    )
    parser.add_argument(
        "--live-csv",
        type=Path,
        default=BASE_DIR / "live_features.csv",
        help="CSV written by live_intrusion_detector.py",
    )
    args = parser.parse_args()

    model_path = (
        ADAPTED_MODEL_PATH
        if ADAPTED_MODEL_PATH.exists()
        else MODEL_PATH
    )
    model = joblib.load(model_path)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print("Model file:", model_path)
    print("Model classes:", list(model.classes_))
    print("Model feature order:", list(model.feature_names_in_))
    print("Configured feature order:", FEATURES)
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError("Saved model feature order does not match FEATURES.")

    print("\nTraining labels:")
    print(train["Label"].value_counts().to_string())
    print("\nTest labels:")
    print(test["Label"].value_counts().to_string())
    describe_features(train, "Training")

    test_values, test_invalid = validate(test, "Held-out test")
    if test_invalid.any():
        raise ValueError("Held-out test data contains invalid feature values.")
    test_predictions = model.predict(test_values)
    test_probabilities = model.predict_proba(test_values)
    test_results = pd.DataFrame(
        {
            "actual": test["Label"],
            "predicted": test_predictions,
            "p_anomaly": test_probabilities[
                :, list(model.classes_).index("Anomaly")
            ],
            "p_normal": test_probabilities[
                :, list(model.classes_).index("Normal")
            ],
        }
    )
    print("\nHeld-out test prediction counts:")
    print(test_results["predicted"].value_counts().to_string())
    print("\nHeld-out test results by actual label:")
    print(
        test_results.groupby("actual")["predicted"]
        .value_counts()
        .to_string()
    )

    if not args.live_csv.exists():
        print(f"\nLive CSV not found: {args.live_csv}")
        print("Run live_intrusion_detector.py first; no packet capture is "
              "started by this diagnostic.")
        return

    live = pd.read_csv(args.live_csv)
    missing = [column for column in FEATURES if column not in live.columns]
    if missing:
        raise ValueError(f"Live CSV is missing features: {missing}")
    describe_features(live, "Live")
    live_values, live_invalid = validate(live, "Live")
    if live_invalid.any():
        print("Live predictions skipped because invalid rows are present.")
        return

    probabilities = model.predict_proba(live_values)
    anomaly_index = list(model.classes_).index("Anomaly")
    normal_index = list(model.classes_).index("Normal")
    output = live.copy()
    output["prediction"] = model.predict(live_values)
    output["p_anomaly"] = probabilities[:, anomaly_index]
    output["p_normal"] = probabilities[:, normal_index]
    print("\nLive prediction counts:")
    print(output["prediction"].value_counts().to_string())
    print("\nLive confidence summary:")
    print(
        output[["p_anomaly", "p_normal"]]
        .describe(percentiles=[0.01, 0.50, 0.99])
        .T[["min", "1%", "50%", "99%", "max", "mean"]]
        .round(4)
        .to_string()
    )
    print("\nFirst live predictions:")
    print(
        output[
            ["prediction", "p_anomaly", "p_normal", "Protocol",
             "Src_Port", "Dst_Port", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
