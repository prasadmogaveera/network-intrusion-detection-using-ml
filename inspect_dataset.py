
from pathlib import Path
import pandas as pd
import numpy as np

file_path = Path("IoT Network Intrusion Dataset.csv")

if not file_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {file_path.resolve()}"
    )

print("File size (MB):", round(
    file_path.stat().st_size / (1024 * 1024), 2
))

df = pd.read_csv(file_path, low_memory=False)

print("\nDataset shape:", df.shape)

print("\nColumn names:")
for i, col in enumerate(df.columns, start=1):
    print(i, repr(col))

print("\nData types:")
print(df.dtypes.to_string())

print("\nFirst five rows:")
print(df.head().to_string())

print("\nPossible label columns:")
label_candidates = [
    col for col in df.columns
    if any(word in str(col).lower()
           for word in ["label", "class", "attack", "category", "target"])
]

print(label_candidates)

for col in label_candidates:
    print(f"\nValue counts for {col!r}:")
    print(df[col].value_counts(dropna=False).to_string())

print("\nMissing values:")
missing = df.isna().sum()
print(missing[missing > 0].to_string()
      if missing.any() else "No missing values found.")

print("\nExact duplicate rows:", df.duplicated().sum())

numeric_df = df.select_dtypes(include=[np.number])

print("\nInfinite values by column:")
if numeric_df.empty:
    print("No numeric columns found.")
else:
    infinite_counts = pd.Series(
        np.isinf(numeric_df.to_numpy()).sum(axis=0),
        index=numeric_df.columns
    )
    print(infinite_counts[infinite_counts > 0].to_string()
          if (infinite_counts > 0).any()
          else "No infinite values found.")

print("\nNumeric summary:")
print(numeric_df.describe().T.to_string())

print("\nInspection completed. No model was trained.")
