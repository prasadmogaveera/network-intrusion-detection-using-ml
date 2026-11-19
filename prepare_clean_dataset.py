
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_FILE = Path("IoT Network Intrusion Dataset.csv")

BINARY_OUTPUT = Path("IoT_Network_Intrusion_Clean_Binary.csv")
MULTICLASS_OUTPUT = Path("IoT_Network_Intrusion_Clean_Multiclass.csv")
REPORT_FILE = Path("dataset_cleaning_report.txt")

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

TARGETS = ["Label", "Cat"]


# --------------------------------------------------
# LOAD AND VALIDATE
# --------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE.resolve()}\n"
        "Place this script in the same folder as the CSV."
    )

print("=" * 60)
print("IOT INTRUSION DATASET CLEANING")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

original_rows = len(df)
original_columns = len(df.columns)

print(f"Original rows: {original_rows:,}")
print(f"Original columns: {original_columns}")

required = FEATURES + TARGETS
missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

report = []
report.append("IOT INTRUSION DATASET CLEANING REPORT")
report.append("=" * 50)
report.append(f"Input file: {INPUT_FILE}")
report.append(f"Original rows: {original_rows:,}")
report.append(f"Original columns: {original_columns}")


# --------------------------------------------------
# STEP 1: REMOVE EXACT DUPLICATE RECORDS
# --------------------------------------------------

before = len(df)
df = df.drop_duplicates().copy()
exact_duplicates = before - len(df)

print(f"\nExact duplicate rows removed: {exact_duplicates:,}")

report.append(f"Exact duplicate rows removed: {exact_duplicates:,}")
report.append(f"Rows after exact deduplication: {len(df):,}")


# --------------------------------------------------
# STEP 2: CONVERT SELECTED FEATURES TO NUMERIC
# --------------------------------------------------

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in TARGETS:
    df[col] = df[col].astype("string").str.strip()

# Identify rows with missing or non-finite model features.
feature_values = df[FEATURES].to_numpy(dtype=float)
valid_features = np.isfinite(feature_values).all(axis=1)

invalid_feature_rows = int((~valid_features).sum())
df = df.loc[valid_features].copy()

print(f"Rows with invalid candidate features removed: "
      f"{invalid_feature_rows:,}")

report.append(
    f"Rows with missing/non-finite candidate features removed: "
    f"{invalid_feature_rows:,}"
)


# --------------------------------------------------
# STEP 3: VALIDATE TARGET LABELS
# --------------------------------------------------

valid_labels = {"Normal", "Anomaly"}
valid_categories = {
    "Normal",
    "Mirai",
    "Scan",
    "DoS",
    "MITM ARP Spoofing",
}

invalid_binary = ~df["Label"].isin(valid_labels)
invalid_multiclass = ~df["Cat"].isin(valid_categories)

print(f"Rows with invalid binary labels: "
      f"{int(invalid_binary.sum()):,}")

print(f"Rows with invalid multiclass labels: "
      f"{int(invalid_multiclass.sum()):,}")

report.append(
    f"Rows with invalid binary labels: {int(invalid_binary.sum()):,}"
)
report.append(
    "Rows with invalid multiclass labels: "
    f"{int(invalid_multiclass.sum()):,}"
)


# --------------------------------------------------
# STEP 4: IDENTIFY AMBIGUOUS FEATURE VECTORS
# --------------------------------------------------

# Identical feature vectors with different target labels
# cannot be distinguished by a model using these features.
# Exclude those vectors from the relevant output only.

def create_clean_output(target, valid_target_values, output_path):
    target_df = df.loc[
        df[target].isin(valid_target_values)
    ].copy()

    rows_before_conflicts = len(target_df)

    # Hash only the model features, not identifiers or labels.
    feature_hash = pd.util.hash_pandas_object(
        target_df[FEATURES],
        index=False
    )

    label_counts = target_df.groupby(feature_hash)[target].nunique()
    conflicting_hashes = label_counts[label_counts > 1].index

    conflicting_mask = feature_hash.isin(conflicting_hashes)
    conflicting_rows = int(conflicting_mask.sum())

    target_df = target_df.loc[~conflicting_mask].copy()

    # Keep only the approved features and this target.
    output_df = target_df[FEATURES + [target]].copy()

    # Final safety checks.
    if output_df[FEATURES].isna().any().any():
        raise ValueError(f"Missing features remain in {target} output.")

    if not np.isfinite(
        output_df[FEATURES].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Non-finite features remain in {target} output.")

    if output_df[target].isna().any():
        raise ValueError(f"Missing targets remain in {target} output.")

    output_df.to_csv(output_path, index=False)

    print(f"\n--- {target} DATASET ---")
    print(f"Rows before conflict filtering: {rows_before_conflicts:,}")
    print(f"Ambiguous feature-vector rows removed: {conflicting_rows:,}")
    print(f"Final rows: {len(output_df):,}")
    print(f"Saved: {output_path}")

    print("\nClass distribution:")
    print(output_df[target].value_counts().to_string())

    report.append("")
    report.append(f"{target} DATASET")
    report.append("-" * 40)
    report.append(
        f"Rows before conflict filtering: {rows_before_conflicts:,}"
    )
    report.append(
        f"Ambiguous feature-vector rows removed: {conflicting_rows:,}"
    )
    report.append(f"Final rows: {len(output_df):,}")
    report.append(f"Output file: {output_path}")
    report.append("Class distribution:")
    report.append(output_df[target].value_counts().to_string())

    return output_df


binary_df = create_clean_output(
    target="Label",
    valid_target_values=valid_labels,
    output_path=BINARY_OUTPUT,
)

multiclass_df = create_clean_output(
    target="Cat",
    valid_target_values=valid_categories,
    output_path=MULTICLASS_OUTPUT,
)


# --------------------------------------------------
# STEP 5: SAVE AUDIT REPORT
# --------------------------------------------------

report.extend([
    "",
    "IMPORTANT NOTES",
    "-" * 40,
    "The original CSV was not modified.",
    "Only exact duplicate full records were removed initially.",
    "Rows with non-finite selected model features were removed.",
    "Ambiguous feature vectors were excluded separately per target.",
    "Zero-duration flows were retained when their selected features were valid.",
    "Unselected infinite rate columns were not used as model features.",
    "IP addresses, timestamps, flow IDs and Sub_Cat were excluded.",
    "No scaling, balancing, imputation or model training was performed.",
    "Cleaning does not guarantee that the data is error-free.",
])

REPORT_FILE.write_text("\n".join(report), encoding="utf-8")

print("\n" + "=" * 60)
print("CLEANING FINISHED")
print("=" * 60)
print(f"Binary dataset: {BINARY_OUTPUT}")
print(f"Multiclass dataset: {MULTICLASS_OUTPUT}")
print(f"Audit report: {REPORT_FILE}")
print("\nOriginal dataset remains unchanged.")
print("Do not train yet; review the report first.")
