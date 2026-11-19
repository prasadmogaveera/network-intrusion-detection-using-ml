import pandas as pd

file_path = "IoT Network Intrusion Dataset.csv"

df = pd.read_csv(file_path, nrows=5)

print("\nColumns related to time:")
for col in df.columns:
    if any(word in col.lower() for word in
           ["time", "timestamp", "src_ip", "dst_ip", "flow_id"]):
        print(col)

print("\nFirst 5 rows:")
print(df.head())