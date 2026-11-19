"""
Generate project figures for the IoT Network Intrusion Detection System.

Run:
    python generate_project_figures.py

Output:
    project_figures/

This script generates data-driven figures where the required files are available.
It does NOT invent model scores. Confusion matrices, ROC/PR curves, and model
comparisons require a compatible saved test set and model artifacts.
"""

from pathlib import Path
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "project_figures"
CLEAN_DATASET_CANDIDATES = [
    BASE_DIR / "IoT_Network_Intrusion_Clean_Binary.csv",
    BASE_DIR / "IoT Network Intrusion Dataset.csv",
]
MULTICLASS_DATASET = BASE_DIR / "IoT_Network_Intrusion_Clean_Multiclass.csv"
FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration", "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts", "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts", "Pkt_Len_Min",
    "Pkt_Len_Max", "Pkt_Len_Mean", "Pkt_Len_Std", "Flow_IAT_Mean",
    "Flow_IAT_Std", "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt",
]
LABEL_COL = "Label"
SEED = 42


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[OK] {filename}")


def find_dataset():
    for path in CLEAN_DATASET_CANDIDATES:
        if path.exists():
            return path
    # Look for likely cleaned/binary CSVs, but avoid live output files.
    for path in BASE_DIR.glob("*.csv"):
        name = path.name.lower()
        if "clean" in name and "binary" in name and "live" not in name:
            return path
    return None


def normalize_labels(y):
    s = pd.Series(y).astype(str).str.strip()
    return s.replace({
        "0": "Normal", "1": "Anomaly",
        "normal": "Normal", "anomaly": "Anomaly",
        "NORMAL": "Normal", "ANOMALY": "Anomaly",
    })


def make_architecture():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.3, 3.6, 2.0, 1.2, "Network traffic\n(Npcap on Windows)"),
        (2.7, 3.6, 2.0, 1.2, "Packet capture\n(Scapy)"),
        (5.1, 3.6, 2.0, 1.2, "Flow aggregation\n+ 19 features"),
        (7.5, 3.6, 1.8, 1.2, "Trained ML\nclassifier"),
        (9.7, 3.6, 2.0, 1.2, "Streamlit\nDashboard"),
    ]
    for x, y, w, h, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)
    for i in range(len(boxes) - 1):
        x, y, w, h, _ = boxes[i]
        nx, ny, _, nh, _ = boxes[i + 1]
        ax.annotate("", xy=(nx - 0.08, ny + nh / 2), xytext=(x + w + 0.08, y + h / 2),
                    arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.text(6, 2.35, "Saved outputs: live_features.csv  •  live_predictions.csv",
            ha="center", fontsize=11)
    ax.text(6, 1.65, "Monitoring prototype — predictions are indicators, not confirmed attacks",
            ha="center", fontsize=10)
    ax.set_title("IoT Network Intrusion Detection System — Architecture", fontsize=15, pad=15)
    save_plot("01_system_architecture.png")


def make_class_distribution(dataset):
    if LABEL_COL not in dataset.columns:
        print("[SKIP] Class distribution: no 'Label' column found.")
        return
    counts = dataset[LABEL_COL].astype(str).value_counts()
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        counts.index,
        counts.values,
        color=["#d95f02" if value == "Anomaly" else "#1b9e77"
               for value in counts.index],
    )
    plt.title("IoT Network Intrusion Dataset — Binary Classification")
    plt.xlabel("Class")
    plt.ylabel("Number of records")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:,}", ha="center", va="bottom", fontsize=9)
    save_plot("02_binary_dataset_classification.png")


def make_multiclass_distribution(dataset_path):
    if not dataset_path.exists():
        print("[SKIP] Multiclass distribution: cleaned multiclass dataset not found.")
        return
    dataset = pd.read_csv(dataset_path, usecols=["Cat"])
    counts = dataset["Cat"].astype(str).str.strip().value_counts()
    plt.figure(figsize=(10, 5.5))
    bars = plt.bar(counts.index, counts.values, color=plt.cm.Set2.colors[:len(counts)])
    plt.title("IoT Network Intrusion Dataset — Multiclass Classification")
    plt.xlabel("Traffic class")
    plt.ylabel("Number of records")
    plt.xticks(rotation=20, ha="right")
    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    save_plot("04_dataset_multiclass_classification.png")


def make_balanced_binary_distribution(dataset_path):
    if not dataset_path.exists():
        print("[SKIP] Balanced binary distribution: dataset not found.")
        return
    dataset = pd.read_csv(dataset_path, usecols=[LABEL_COL])
    dataset[LABEL_COL] = dataset[LABEL_COL].astype(str).str.strip()
    class_counts = dataset[LABEL_COL].value_counts()
    if len(class_counts) != 2:
        print("[SKIP] Balanced binary distribution: expected two classes.")
        return
    target_count = int(class_counts["Anomaly"] // 2)
    balanced = pd.concat(
        [
            dataset[dataset[LABEL_COL] == label].sample(
                n=target_count,
                replace=len(dataset[dataset[LABEL_COL] == label]) < target_count,
                random_state=SEED,
            )
            for label in class_counts.index
        ]
    )
    counts = balanced[LABEL_COL].value_counts().reindex(
        ["Normal", "Anomaly"], fill_value=0
    )
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        counts.index,
        counts.values,
        color=["#1b9e77", "#d95f02"],
    )
    plt.title("Balanced Binary Dataset (211,647 Records per Class)")
    plt.xlabel("Class")
    plt.ylabel("Number of records")
    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save_plot("05_balanced_binary_dataset_classification.png")


def make_pcc(dataset):
    available = [c for c in FEATURES if c in dataset.columns]
    if LABEL_COL not in dataset.columns or len(available) < 2:
        print("[SKIP] PCC: dataset lacks numeric features or Label.")
        return
    # Binary label correlation: Normal=0, Anomaly=1; only for binary labels.
    labels = dataset[LABEL_COL].astype(str).str.strip().str.lower()
    binary_map = {"normal": 0, "anomaly": 1}
    if not set(labels.dropna().unique()).issubset(set(binary_map)):
        print("[SKIP] PCC: found non-binary labels; not silently converting multiclass labels.")
        return
    temp = dataset[available].apply(pd.to_numeric, errors="coerce")
    temp["Binary_Label"] = labels.map(binary_map)
    corr = temp.corr(numeric_only=True)["Binary_Label"].drop("Binary_Label").dropna()
    if corr.empty:
        print("[SKIP] PCC: correlations could not be calculated.")
        return
    corr = corr.reindex(corr.abs().sort_values(ascending=True).index)
    plt.figure(figsize=(10, 7))
    plt.barh(corr.index, corr.values)
    plt.axvline(0, linewidth=0.8)
    plt.title("Pearson Correlation Coefficient (PCC) with Binary Label")
    plt.xlabel("Pearson correlation (r)")
    save_plot("03_pcc_feature_correlation.png")


def find_test_data():
    # Accept common naming styles; user can add their actual filenames here if needed.
    x_candidates = [
        "X_test_binary.csv", "X_test.csv", "X_test_bin.csv",
        "X_test_binary_split.csv", "X_test_features.csv",
    ]
    y_candidates = [
        "y_test_binary.csv", "y_test.csv", "y_test_bin.csv",
        "y_test_binary_split.csv", "y_test_labels.csv",
    ]
    x_path = next((BASE_DIR / n for n in x_candidates if (BASE_DIR / n).exists()), None)
    y_path = next((BASE_DIR / n for n in y_candidates if (BASE_DIR / n).exists()), None)
    if x_path and y_path:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
        # A single-column y file is expected; use its first column.
        return X, y.iloc[:, 0], f"{x_path.name} + {y_path.name}"
    return None, None, None


def try_load_models():
    models = {}
    try:
        import joblib
        for name, filename in [
            ("Random Forest", "random_forest_binary.joblib"),
            ("XGBoost", "xgboost_binary.joblib"),
        ]:
            path = BASE_DIR / filename
            if path.exists():
                try:
                    models[name] = ("sklearn", joblib.load(path), None)
                except Exception as exc:
                    print(f"[SKIP] Could not load {filename}: {exc}")
    except ImportError:
        print("[INFO] joblib is not installed; skipping joblib models.")

    ann_path = BASE_DIR / "ann_binary.keras"
    if ann_path.exists():
        try:
            from tensorflow import keras
            model = keras.models.load_model(ann_path)
            scaler_path = BASE_DIR / "ann_scaler.joblib"
            scaler = None
            if scaler_path.exists():
                import joblib
                scaler = joblib.load(scaler_path)
            models["ANN"] = ("keras", model, scaler)
        except Exception as exc:
            print(f"[SKIP] Could not load ANN: {exc}")

    # LSTM is not automatically evaluated: its sequence construction and feature
    # format may differ from the tabular models, so a generic comparison could mislead.
    if (BASE_DIR / "lstm_binary.keras").exists() or (BASE_DIR / "lstm_model.keras").exists():
        print("[INFO] LSTM model found, but skipped because its sequence preprocessing must be matched explicitly.")
    return models


def predict(model_info, X):
    kind, model, scaler = model_info
    if kind == "keras":
        values = scaler.transform(X) if scaler is not None else X
        raw = np.asarray(model.predict(values, verbose=0))
        if raw.ndim == 2 and raw.shape[1] == 1:
            p1 = raw[:, 0]
        elif raw.ndim == 2 and raw.shape[1] == 2:
            p1 = raw[:, 1]
        else:
            raise ValueError(f"Unsupported Keras output shape: {raw.shape}")
        return (p1 >= 0.5).astype(int), p1
    pred = model.predict(X)
    classes = list(model.classes_)
    # Map predicted class to binary anomaly=1, normal=0 without assuming class order.
    pred_text = normalize_labels(pred)
    yhat = (pred_text.str.lower() == "anomaly").astype(int).to_numpy()
    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        anomaly_idx = next((i for i, c in enumerate(classes)
                            if str(c).strip().lower() in ("anomaly", "1")), None)
        if anomaly_idx is not None:
            probabilities = probs[:, anomaly_idx]
    return yhat, probabilities


def make_model_figures(X, y, source):
    from sklearn.metrics import (
        confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
        precision_recall_curve, average_precision_score, accuracy_score,
        precision_score, recall_score, f1_score,
    )

    y_norm = normalize_labels(y)
    valid = y_norm.str.lower().isin(["normal", "anomaly"])
    X = X.loc[valid].copy()
    y_true = (y_norm.loc[valid].str.lower() == "anomaly").astype(int).to_numpy()
    missing = [f for f in FEATURES if f not in X.columns]
    if missing:
        print(f"[SKIP] Model evaluation: test set missing required features: {missing}")
        return
    X = X[FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_rows = X.notna().all(axis=1).to_numpy()
    X = X.loc[valid_rows]
    y_true = y_true[valid_rows]
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        print("[SKIP] Model evaluation: test set needs both Normal and Anomaly rows.")
        return
    print(f"[INFO] Evaluating on {len(y_true):,} valid rows from {source}.")
    models = try_load_models()
    metric_rows, roc_entries, pr_entries = [], [], []
    for name, model_info in models.items():
        try:
            start = time.perf_counter()
            y_pred, score = predict(model_info, X)
            elapsed = time.perf_counter() - start
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(5.8, 5))
            ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"]).plot(
                ax=ax, cmap="Blues", colorbar=False
            )
            ax.set_title(f"{name} — Confusion Matrix")
            save_plot(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")

            metric_rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Precision (Anomaly)": precision_score(y_true, y_pred, zero_division=0),
                "Recall (Anomaly)": recall_score(y_true, y_pred, zero_division=0),
                "F1 (Anomaly)": f1_score(y_true, y_pred, zero_division=0),
                "Prediction time (s)": elapsed,
            })
            if score is not None and len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, score)
                roc_entries.append((name, fpr, tpr, auc(fpr, tpr)))
                precision, recall, _ = precision_recall_curve(y_true, score)
                pr_entries.append((name, recall, precision, average_precision_score(y_true, score)))
        except Exception as exc:
            print(f"[SKIP] {name} evaluation failed: {exc}")

    if metric_rows:
        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(OUTPUT_DIR / "model_metrics_generated.csv", index=False)
        plt.figure(figsize=(10, 5.5))
        x = np.arange(len(metrics))
        width = 0.22
        for j, col in enumerate(["Accuracy", "Precision (Anomaly)", "Recall (Anomaly)", "F1 (Anomaly)"]):
            plt.bar(x + (j - 1.5) * width, metrics[col], width, label=col)
        plt.xticks(x, metrics["Model"])
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Model Performance Comparison (computed on supplied test set)")
        plt.legend()
        save_plot("model_performance_comparison.png")

        plt.figure(figsize=(8, 5.5))
        plt.bar(metrics["Model"], metrics["Prediction time (s)"])
        plt.ylabel("Prediction time on this run (seconds)")
        plt.title("Model Prediction-Time Comparison")
        save_plot("model_prediction_time_comparison.png")

    if roc_entries:
        plt.figure(figsize=(7.5, 6))
        for name, fpr, tpr, area in roc_entries:
            plt.plot(fpr, tpr, label=f"{name} (AUC={area:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend()
        save_plot("roc_curve_comparison.png")

    if pr_entries:
        plt.figure(figsize=(7.5, 6))
        for name, recall, precision, ap in pr_entries:
            plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve Comparison")
        plt.legend()
        save_plot("precision_recall_curve.png")


def make_rf_importance():
    try:
        import joblib
        model_path = BASE_DIR / "random_forest_binary.joblib"
        if not model_path.exists():
            print("[SKIP] RF feature importance: random_forest_binary.joblib not found.")
            return
        model = joblib.load(model_path)
        if not hasattr(model, "feature_importances_"):
            print("[SKIP] Loaded Random Forest does not expose feature_importances_.")
            return
        names = list(getattr(model, "feature_names_in_", FEATURES))
        values = model.feature_importances_
        if len(names) != len(values):
            names = [f"Feature {i+1}" for i in range(len(values))]
        order = np.argsort(values)
        plt.figure(figsize=(10, 7))
        plt.barh(np.array(names)[order], values[order])
        plt.xlabel("Feature importance")
        plt.title("Random Forest Feature Importance")
        save_plot("random_forest_feature_importance.png")
    except Exception as exc:
        print(f"[SKIP] RF feature importance: {exc}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {OUTPUT_DIR}")
    make_architecture()

    dataset_path = find_dataset()
    dataset = None
    if dataset_path:
        try:
            dataset = pd.read_csv(dataset_path, low_memory=False)
            print(f"[INFO] Dataset loaded: {dataset_path.name} ({len(dataset):,} rows)")
            make_class_distribution(dataset)
            make_pcc(dataset)
        except Exception as exc:
            print(f"[WARN] Could not read dataset: {exc}")
    else:
        print("[INFO] No recognized dataset CSV found; class distribution/PCC skipped.")

    make_multiclass_distribution(MULTICLASS_DATASET)
    make_balanced_binary_distribution(BASE_DIR / "IoT_Network_Intrusion_Clean_Binary.csv")
    make_rf_importance()
    X_test, y_test, source = find_test_data()
    if X_test is not None:
        make_model_figures(X_test, y_test, source)
    else:
        print(
            "[INFO] Compatible X_test/y_test CSV pair not found. Skipped confusion matrices, "
            "ROC, PR, performance and prediction-time charts rather than inventing results. "
            "Add your actual test split filenames to find_test_data() in this script."
        )

    print("\\nDone. Review the project_figures folder and the messages above.")
    print("Note: training-time comparison requires recorded training durations; this script does not invent them.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
