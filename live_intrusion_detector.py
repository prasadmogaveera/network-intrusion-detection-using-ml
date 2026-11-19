
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flow_aggregator_v5 import (
    find_interface,
    process_packet,
    summarize,
    finish_flow,
    flows,
    completed_flows,
    sniff,
    CAPTURE_SECONDS,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_binary.joblib"
ADAPTED_MODEL_PATH = BASE_DIR / "random_forest_live_adapted.joblib"
LIVE_FEATURES_PATH = BASE_DIR / "live_features.csv"
LIVE_PREDICTIONS_PATH = BASE_DIR / "live_predictions.csv"

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


def to_model_features(flow):
    result = {
        "Protocol": flow["protocol"],
        "Src_Port": flow["Src_Port"],
        "Dst_Port": flow["Dst_Port"],
        "Flow_Duration": flow["duration_us"],
        "Tot_Fwd_Pkts": flow["fwd_packets"],
        "Tot_Bwd_Pkts": flow["bwd_packets"],
        "TotLen_Fwd_Pkts": flow["fwd_bytes"],
        "TotLen_Bwd_Pkts": flow["bwd_bytes"],
        "Pkt_Len_Min": flow["packet_length_min"],
        "Pkt_Len_Max": flow["packet_length_max"],
        "Pkt_Len_Mean": flow["packet_length_mean"],
        "Pkt_Len_Std": flow["packet_length_std"],
        "Flow_IAT_Mean": flow["mean_iat_us"],
        "Flow_IAT_Std": flow["std_iat_us"],
        "SYN_Flag_Cnt": flow.get("SYN", 0),
        "ACK_Flag_Cnt": flow.get("ACK", 0),
        "RST_Flag_Cnt": flow.get("RST", 0),
        "FIN_Flag_Cnt": flow.get("FIN", 0),
        "PSH_Flag_Cnt": flow.get("PSH", 0),
    }

    if (
        result["Tot_Fwd_Pkts"] + result["Tot_Bwd_Pkts"]
        != flow["total_packets"]
    ):
        raise ValueError("Directional packet totals do not match flow total.")

    if (
        result["TotLen_Fwd_Pkts"] + result["TotLen_Bwd_Pkts"]
        != flow["total_bytes"]
    ):
        raise ValueError("Directional byte totals do not match flow total.")

    return result

def main():
    # Load the trained model.
    model_path = (
        ADAPTED_MODEL_PATH
        if ADAPTED_MODEL_PATH.exists()
        else MODEL_PATH
    )
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    model = joblib.load(model_path)

    print("LIVE INTRUSION DETECTOR — PROTOTYPE")
    print("Model file:", model_path)
    print("Model classes:", model.classes_)

    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        print("Expected features:", list(expected_features))

        if list(expected_features) != FEATURES:
            print("\nERROR: Model feature order does not match.")
            print("Expected:", list(expected_features))
            print("Provided:", FEATURES)
            return

    interface = find_interface()

    if interface is None:
        print("Network interface not found.")
        return

    print("Interface:", interface)
    print("Capture duration:", CAPTURE_SECONDS, "seconds")
    print("Monitoring only; no traffic will be blocked.")
    print("Predictions are experimental, not confirmed attack detections.")

    input("\nPress Enter to start capture...")

    # Clear results from any previous run in this Python process.
    flows.clear()
    completed_flows.clear()

    try:
        sniff(
            iface=interface,
            timeout=CAPTURE_SECONDS,
            store=False,
            prn=process_packet,
        )
    except PermissionError:
        print(
            "Permission denied. Run the terminal with suitable "
            "packet-capture permissions."
        )
        return
    except Exception as exc:
        print("Capture failed:", exc)
        return

    # Finalize active flows using the aggregator's own logic.
    for key in list(flows):
        finish_flow(key)

    records = [summarize(flow) for flow in completed_flows]

    if not records:
        print("\nNo flows captured. Try again while browsing websites.")
        return

    # Build features in the exact order used by the trained model.
    rows = pd.DataFrame(
        [to_model_features(flow) for flow in records],
        columns=FEATURES,
    )

    rows = rows.replace([np.inf, -np.inf], np.nan)

    if rows.isna().any().any():
        invalid_count = int(rows.isna().any(axis=1).sum())
        print(
            f"\nERROR: {invalid_count} flow(s) contain missing or "
            "infinite feature values."
        )
        print("Predictions stopped to avoid unreliable input.")
        return

    rows.to_csv(LIVE_FEATURES_PATH, index=False)
    print("\nSaved live feature rows:", LIVE_FEATURES_PATH)

    # Check that the model accepts the feature matrix.
    try:
        predictions = model.predict(rows)
        probabilities = model.predict_proba(rows)
    except Exception as exc:
        print("\nPrediction failed:", exc)
        return

    class_indices = {label: index for index, label in enumerate(model.classes_)}
    prediction_output = rows.copy()
    prediction_output["Prediction"] = predictions
    prediction_output["P_Anomaly"] = probabilities[
        :, class_indices["Anomaly"]
    ]
    prediction_output["P_Normal"] = probabilities[
        :, class_indices["Normal"]
    ]
    prediction_output.to_csv(LIVE_PREDICTIONS_PATH, index=False)
    print("Saved live predictions:", LIVE_PREDICTIONS_PATH)

    print("\nCompleted flows:", len(records))

    print("\nPrediction counts:")
    print(pd.Series(predictions).value_counts().to_string())

    print("\nPrediction probabilities by model class:")
    print(
        pd.DataFrame(
            probabilities,
            columns=[f"P({label})" for label in model.classes_],
        )
        .describe()
        .T[["min", "mean", "max"]]
        .to_string()
    )

    print("\nFirst 5 feature rows sent to the model:")
    print(rows.head().to_string(index=False))

    paired_output = rows[
        [
            "Tot_Fwd_Pkts",
            "Tot_Bwd_Pkts",
            "TotLen_Fwd_Pkts",
            "TotLen_Bwd_Pkts",
        ]
    ].copy()
    paired_output.insert(
        0,
        "Flow",
        range(1, len(paired_output) + 1),
    )
    paired_output["Total_Packets"] = (
        paired_output["Tot_Fwd_Pkts"]
        + paired_output["Tot_Bwd_Pkts"]
    )
    paired_output["Summary_Packets"] = [
        flow["total_packets"] for flow in records
    ]
    paired_output["Total_Bytes"] = (
        paired_output["TotLen_Fwd_Pkts"]
        + paired_output["TotLen_Bwd_Pkts"]
    )
    paired_output["Summary_Bytes"] = [
        flow["total_bytes"] for flow in records
    ]
    paired_output["Prediction"] = predictions
    print("\nVerified feature-to-flow mapping:")
    print(paired_output.head(10).to_string(index=False))

    print("\nFeature ranges:")
    print(
        rows.describe()
        .T[["min", "mean", "max"]]
        .to_string()
    )

    print("\nLIVE FLOW PREDICTIONS")
    print("=" * 60)

    for index, (flow, prediction) in enumerate(
        zip(records, predictions), start=1
    ):
        print(
            f"Flow {index}: {prediction} | "
            f"Protocol={flow['protocol']} | "
            f"Packets={flow['total_packets']} | "
            f"Bytes={flow['total_bytes']}"
        )

    print("\nPrototype run completed.")
    print("No traffic was blocked.")
    print("Anomaly predictions are not proof of an attack.")


if __name__ == "__main__":
    main()
