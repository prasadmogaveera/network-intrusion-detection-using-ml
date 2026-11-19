
import pandas as pd
import numpy as np

file_path = "IoT Network Intrusion Dataset.csv"

df = pd.read_csv(file_path, low_memory=False)

# 1. Inspect all three label columns
for col in ["Label", "Cat", "Sub_Cat"]:
    print("\n" + "=" * 60)
    print(f"VALUE COUNTS: {col}")
    print("=" * 60)

    print(df[col].value_counts(dropna=False).to_string())
    print("Unique values:", df[col].nunique(dropna=False))

# 2. Check how categories correspond to binary labels
print("\n" + "=" * 60)
print("LABEL VS ATTACK CATEGORY")
print("=" * 60)

print(pd.crosstab(
    df["Label"],
    df["Cat"],
    dropna=False
).to_string())

# 3. Find columns that are constant
print("\n" + "=" * 60)
print("CONSTANT COLUMNS")
print("=" * 60)

constant_cols = [
    col for col in df.columns
    if df[col].nunique(dropna=False) <= 1
]

print(constant_cols)

# 4. Find columns where at least 99.9% of values are zero
print("\n" + "=" * 60)
print("COLUMNS THAT ARE ALMOST ALWAYS ZERO")
print("=" * 60)

numeric_df = df.select_dtypes(include=[np.number])

for col in numeric_df.columns:
    zero_ratio = numeric_df[col].eq(0).mean()

    if zero_ratio >= 0.999:
        print(f"{col}: {zero_ratio * 100:.4f}% zero")

# 5. Inspect infinite values and their affected rows
print("\n" + "=" * 60)
print("INFINITE VALUE DETAILS")
print("=" * 60)

for col in numeric_df.columns:
    values = numeric_df[col].to_numpy()
    count = np.isinf(values).sum()

    if count:
        print(f"\n{col}: {count} infinite values")
        print("Affected label counts:")
        print(df.loc[np.isinf(values), "Label"].value_counts())

# 6. Inspect duplicate patterns without deleting anything
print("\n" + "=" * 60)
print("DUPLICATE INVESTIGATION")
print("=" * 60)

print("Exact duplicate rows:", df.duplicated().sum())

# Compare duplicates using flow features only,
# excluding identifiers, timestamp and labels.
excluded = [
    "Flow_ID", "Src_IP", "Dst_IP", "Timestamp",
    "Label", "Cat", "Sub_Cat"
]

feature_cols = [col for col in df.columns if col not in excluded]

print(
    "Rows duplicated across the remaining feature columns:",
    df.duplicated(subset=feature_cols).sum()
)

# 7. Check whether labels conflict for identical feature vectors
label_check = df.groupby(feature_cols, dropna=False)[
    ["Label", "Cat", "Sub_Cat"]
].nunique(dropna=False)

print(
    "Unique feature groups with conflicting binary labels:",
    (label_check["Label"] > 1).sum()
)

print(
    "Unique feature groups with conflicting attack categories:",
    (label_check["Cat"] > 1).sum()
)

print("\nInspection complete. No rows or columns were modified.")

