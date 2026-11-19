
import pandas as pd
import numpy as np

df = pd.read_csv("IoT Network Intrusion Dataset.csv")

zero = df[df["Flow_Duration"] == 0]
nonzero = df[df["Flow_Duration"] > 0]

print("\n1. ZERO-DURATION FLOWS")
print("=" * 50)
print("Total:", len(zero))
print("\nCategories:")
print(zero["Cat"].value_counts().to_string())

print("\nSubcategories:")
print(zero["Sub_Cat"].value_counts().to_string())

print("\nProtocols:")
print(zero["Protocol"].value_counts().to_string())

print("\n2. INFINITE RATES IN ZERO-DURATION FLOWS")
print("=" * 50)

for col in ["Flow_Byts/s", "Flow_Pkts/s"]:
    print(
        f"{col}:",
        int(np.isinf(zero[col]).sum())
    )

print("\n3. DURATION COMPARISON")
print("=" * 50)

features = [
    "Tot_Fwd_Pkts",
    "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts",
    "TotLen_Bwd_Pkts",
    "Pkt_Len_Mean",
    "Flow_Duration"
]

print("\nZero-duration median:")
print(zero[features].median(numeric_only=True).to_string())

print("\nNon-zero-duration median:")
print(nonzero[features].median(numeric_only=True).to_string())

print("\n4. ZERO-DURATION EXAMPLES")
print("=" * 50)

cols = features + ["Protocol", "Cat", "Sub_Cat"]
print(zero[cols].head(10).to_string(index=False))

print("\nAudit complete. Original CSV unchanged.")
