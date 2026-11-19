from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from flow_aggregator_v5 import (
    completed_flows,
    find_interface,
    finish_flow,
    flows,
    process_packet,
    sniff,
    summarize,
)
from live_intrusion_detector import to_model_features

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


BASE_DIR = Path(__file__).resolve().parent
ADAPTED_MODEL_PATH = BASE_DIR / "random_forest_live_adapted.joblib"
ORIGINAL_MODEL_PATH = BASE_DIR / "random_forest_binary.joblib"
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


st.set_page_config(
    page_title="IoT Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1220; color: #e5e7eb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stMetric"] {
        background: #172033;
        border: 1px solid #263653;
        border-radius: 12px;
        padding: 12px;
    }
    .app-subtitle { color: #9ca3af; margin-top: -14px; margin-bottom: 24px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def read_csv(path_string):
    return pd.read_csv(path_string)


@st.cache_resource(show_spinner=False)
def load_model(path_string):
    return joblib.load(path_string)


def choose_compatible_model():
    candidates = [
        ("random_forest_live_adapted.joblib", ADAPTED_MODEL_PATH),
        ("random_forest_binary.joblib", ORIGINAL_MODEL_PATH),
    ]
    errors = []
    for filename, path in candidates:
        if not path.exists():
            continue
        try:
            model = load_model(str(path))
            model_features = list(getattr(model, "feature_names_in_", []))
            model_classes = list(getattr(model, "classes_", []))
            if model_features and model_features != FEATURES:
                errors.append(f"{filename}: feature order is incompatible")
                continue
            if model_classes and not {"Anomaly", "Normal"}.issubset(
                set(model_classes)
            ):
                errors.append(f"{filename}: expected Anomaly and Normal classes")
                continue
            return filename, model, errors
        except Exception as error:
            errors.append(f"{filename}: {error}")
    return None, None, errors


def load_live_results():
    if not LIVE_PREDICTIONS_PATH.exists():
        return None, (
            "No live prediction file was found. Run "
            "`live_intrusion_detector.py` first."
        )

    try:
        predictions = read_csv(str(LIVE_PREDICTIONS_PATH))
    except Exception as error:
        return None, f"Could not read live_predictions.csv: {error}"

    if predictions.empty:
        return None, "The live prediction file is empty."
    if "Prediction" not in predictions.columns:
        return None, "The prediction file has no `Prediction` column."

    missing_features = [c for c in FEATURES if c not in predictions.columns]
    if missing_features:
        if not LIVE_FEATURES_PATH.exists():
            return None, (
                "The prediction file is missing feature columns and "
                "live_features.csv is unavailable."
            )
        try:
            features = read_csv(str(LIVE_FEATURES_PATH))
        except Exception as error:
            return None, f"Could not read live_features.csv: {error}"
        if len(features) != len(predictions):
            return None, (
                "The two live files have different row counts; "
                "they cannot be joined safely."
            )
        if any(column not in features.columns for column in FEATURES):
            return None, "live_features.csv does not contain all 19 features."
        predictions = predictions.reset_index(drop=True).copy()
        predictions = pd.concat(
            [features[FEATURES].reset_index(drop=True), predictions],
            axis=1,
        )

    for column in FEATURES:
        predictions[column] = pd.to_numeric(
            predictions[column], errors="coerce"
        )
    if predictions[FEATURES].isna().any().any():
        return None, "The live results contain missing or non-numeric features."

    predictions.insert(0, "Flow", np.arange(1, len(predictions) + 1))
    return predictions, None


def format_duration(value):
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):,.2f} µs ({float(value) / 1_000_000:.6f} s)"


def protocol_name(value):
    names = {6: "TCP (6)", 17: "UDP (17)", 1: "ICMP (1)"}
    try:
        return names.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def display_probability(row):
    if "P_Anomaly" not in row.index or pd.isna(row["P_Anomaly"]):
        return "Unavailable"
    return f"{float(row['P_Anomaly']) * 100:.2f}%"


def capture_and_predict(duration_seconds, model, progress_slot, status_slot):
    interface = find_interface()
    if interface is None:
        return None, (
            "No active network interface was found. Connect to a network or "
            "set CAPTURE_IP to the adapter's IPv4 address."
        )

    flows.clear()
    completed_flows.clear()
    capture_error = []

    def capture():
        try:
            sniff(
                iface=interface,
                timeout=duration_seconds,
                store=False,
                prn=process_packet,
            )
        except Exception as error:
            capture_error.append(error)

    started = monotonic()
    worker = Thread(target=capture, daemon=True)
    worker.start()
    while worker.is_alive():
        elapsed = min(monotonic() - started, duration_seconds)
        progress_slot.progress(
            elapsed / duration_seconds,
            text=f"Capturing traffic... {duration_seconds - elapsed:.1f}s remaining",
        )
        status_slot.caption(
            f"Monitoring interface for {duration_seconds} seconds..."
        )
        sleep(0.2)
    worker.join()

    if capture_error:
        return None, f"Capture failed: {capture_error[0]}"

    for key in list(flows):
        finish_flow(key)
    records = [summarize(flow) for flow in completed_flows]
    records = [record for record in records if record is not None]
    if not records:
        return None, "No TCP or UDP flows were captured."

    try:
        rows = pd.DataFrame(
            [to_model_features(record) for record in records],
            columns=FEATURES,
        ).replace([np.inf, -np.inf], np.nan)
        if rows.isna().any().any():
            return None, "Captured flows contain missing or non-finite features."
        predictions = model.predict(rows)
        probabilities = model.predict_proba(rows)
        class_indices = {
            label: index for index, label in enumerate(model.classes_)
        }
        if not {"Anomaly", "Normal"}.issubset(class_indices):
            return None, "The selected model does not provide both prediction classes."
    except Exception as error:
        return None, f"Prediction failed: {error}"

    prediction_output = rows.copy()
    prediction_output["Prediction"] = predictions
    prediction_output["P_Anomaly"] = probabilities[
        :, class_indices["Anomaly"]
    ]
    prediction_output["P_Normal"] = probabilities[
        :, class_indices["Normal"]
    ]
    rows.to_csv(LIVE_FEATURES_PATH, index=False)
    prediction_output.to_csv(LIVE_PREDICTIONS_PATH, index=False)
    return prediction_output, None


def main():
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "analysis_status" not in st.session_state:
        st.session_state.analysis_status = "Idle — ready for a new capture"

    st.title("🛡️ IoT Network Intrusion Detection System")
    st.markdown(
        '<p class="app-subtitle">Saved-flow monitoring dashboard · '
        "experimental prototype</p>",
        unsafe_allow_html=True,
    )

    _, model, _ = choose_compatible_model()

    st.subheader("Live analysis")
    st.write(
        "Choose a capture duration, click **Start analysis**, and browse "
        "normally while traffic is monitored."
    )
    capture_duration = st.slider(
        "Capture duration (seconds)",
        min_value=5,
        max_value=300,
        value=30,
        step=5,
    )
    st.caption(
        f"Current window: {capture_duration} seconds · "
        f"Status: {st.session_state.analysis_status}"
    )
    start_capture = st.button(
        f"▶ Start analysis ({capture_duration} seconds)",
        type="primary",
        use_container_width=True,
        disabled=model is None,
    )
    if start_capture:
        if model is None:
            st.session_state.analysis_status = "Failed — no compatible model"
            st.error("No compatible Random Forest model is available.")
        else:
            st.session_state.analysis_status = "Running — capturing traffic"
            progress_slot = st.empty()
            status_slot = st.empty()
            with st.spinner("Preparing live analysis..."):
                captured_results, capture_error = capture_and_predict(
                    capture_duration,
                    model,
                    progress_slot,
                    status_slot,
                )
            progress_slot.progress(1.0, text="Capture complete")
            status_slot.empty()
            if capture_error:
                st.session_state.analysis_status = "Failed — capture did not complete"
                st.error(capture_error)
            else:
                read_csv.clear()
                st.session_state.analysis_complete = True
                st.session_state.analysis_status = "Completed — results are ready"
                st.success(
                    f"Analysis complete: {len(captured_results)} flows saved."
                )
                st.rerun()
    st.caption("Monitoring only: no traffic is blocked or modified.")

    if not st.session_state.analysis_complete:
        st.info(
            "No current analysis has been completed. Previous saved results "
            "are hidden. Click **Start analysis** to capture fresh traffic."
        )
        return

    results, load_error = load_live_results()
    if load_error:
        st.warning(load_error)
        st.info(
            "After a successful detector run, click **Refresh saved "
            "results** to load the new CSV files."
        )
        return

    normal_count = int((results["Prediction"] == "Normal").sum())
    anomaly_count = int((results["Prediction"] == "Anomaly").sum())
    total_count = len(results)
    anomaly_percent = anomaly_count / total_count * 100 if total_count else 0

    st.subheader("Dashboard overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Total captured flows", f"{total_count:,}")
    metric_columns[1].metric("Normal", f"{normal_count:,}")
    metric_columns[2].metric("Anomaly", f"{anomaly_count:,}")
    metric_columns[3].metric("Anomaly percentage", f"{anomaly_percent:.2f}%")
    st.caption(
        "These counts describe the currently saved capture only. They are "
        "not a measurement of real-world attack prevalence."
    )

    st.subheader("Prediction visualization")
    chart_columns = st.columns(2)
    chart_data = pd.DataFrame(
        {
            "Prediction": ["Normal", "Anomaly"],
            "Count": [normal_count, anomaly_count],
        }
    )
    with chart_columns[0]:
        st.markdown("**Normal versus Anomaly**")
        if plt is not None:
            figure, axis = plt.subplots(figsize=(5, 3.5))
            axis.pie(
                chart_data["Count"],
                labels=chart_data["Prediction"],
                colors=["#38bdf8", "#ef4444"],
                autopct="%1.1f%%",
                textprops={"color": "white"},
            )
            figure.patch.set_facecolor("#0b1220")
            axis.set_facecolor("#0b1220")
            st.pyplot(figure, clear_figure=True)
        else:
            st.bar_chart(chart_data.set_index("Prediction"), color="#38bdf8")
    with chart_columns[1]:
        st.markdown("**Prediction distribution**")
        st.dataframe(chart_data, hide_index=True, use_container_width=True)

    st.subheader("Live detection results")
    filter_columns = st.columns([2, 1, 1])
    search_text = filter_columns[0].text_input(
        "Search flows",
        placeholder="Search ports, protocol, label, or flow number",
    ).strip().lower()
    label_filter = filter_columns[1].selectbox(
        "Prediction", ["All", "Normal", "Anomaly"]
    )
    protocol_filter = filter_columns[2].selectbox(
        "Protocol",
        ["All"] + sorted(results["Protocol"].dropna().unique().tolist()),
    )

    visible = results.copy()
    if label_filter != "All":
        visible = visible[visible["Prediction"] == label_filter]
    if protocol_filter != "All":
        visible = visible[visible["Protocol"] == protocol_filter]
    if search_text:
        searchable = visible.astype(str).apply(
            lambda column: column.str.lower().str.contains(
                search_text, regex=False, na=False
            )
        )
        visible = visible[searchable.any(axis=1)]

    table_columns = [
        "Flow",
        "Protocol",
        "Src_Port",
        "Dst_Port",
        "Flow_Duration",
        "Tot_Fwd_Pkts",
        "Tot_Bwd_Pkts",
        "TotLen_Fwd_Pkts",
        "TotLen_Bwd_Pkts",
        "Prediction",
    ]
    for optional_column in ["P_Anomaly", "P_Normal"]:
        if optional_column in visible.columns:
            table_columns.append(optional_column)

    def highlight_anomaly(row):
        color = "#4c1d1d" if row.get("Prediction") == "Anomaly" else ""
        return [f"background-color: {color}" if color else "" for _ in row]

    st.caption(f"Showing {len(visible):,} of {total_count:,} saved flows")
    st.dataframe(
        visible[table_columns].style.apply(highlight_anomaly, axis=1),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Flow_Duration": st.column_config.NumberColumn(
                "Duration (µs)", format="%.2f"
            ),
            "P_Anomaly": st.column_config.NumberColumn(
                "P(Anomaly)", format="%.3f"
            ),
            "P_Normal": st.column_config.NumberColumn(
                "P(Normal)", format="%.3f"
            ),
        },
    )

    st.subheader("Flow details")
    if visible.empty:
        st.info("No flows match the current filters.")
    else:
        selected_flow = st.selectbox(
            "Select a flow",
            visible["Flow"].tolist(),
            format_func=lambda value: f"Flow {value}",
        )
        selected = results.loc[results["Flow"] == selected_flow].iloc[0]
        detail_columns = st.columns(4)
        detail_columns[0].metric("Prediction", str(selected["Prediction"]))
        detail_columns[1].metric("P(Anomaly)", display_probability(selected))
        detail_columns[2].metric(
            "Protocol", protocol_name(selected["Protocol"])
        )
        detail_columns[3].metric(
            "Duration", format_duration(selected["Flow_Duration"])
        )
        detail_data = {
            "Source port": selected["Src_Port"],
            "Destination port": selected["Dst_Port"],
            "Forward packets": selected["Tot_Fwd_Pkts"],
            "Backward packets": selected["Tot_Bwd_Pkts"],
            "Forward bytes": selected["TotLen_Fwd_Pkts"],
            "Backward bytes": selected["TotLen_Bwd_Pkts"],
        }
        st.dataframe(
            pd.DataFrame([detail_data]),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "Predictions are experimental indicators, not confirmed attacks. "
        "This tool only monitors traffic and does not block it."
    )


if __name__ == "__main__":
    main()
