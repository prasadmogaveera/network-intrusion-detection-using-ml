
import pandas as pd

original_file = "IoT Network Intrusion Dataset.csv"
clean_file = "IoT_Network_Intrusion_Clean_Binary.csv"

original = pd.read_csv(original_file)
clean = pd.read_csv(clean_file)

print("Original rows:", len(original))
print("Cleaned rows:", len(clean))

print("\nOriginal columns needed for sequence construction:")
print(["Timestamp", "Src_IP", "Dst_IP", "Label"])

print("\nMissing timestamps:", original["Timestamp"].isna().sum())
print("Missing source IPs:", original["Src_IP"].isna().sum())
print("Missing destination IPs:", original["Dst_IP"].isna().sum())

print("\nTimestamp format examples:")
print(original["Timestamp"].head(5).to_list())

print("\nCleaned target distribution:")
print(clean["Label"].value_counts())

print("\nCleaned feature columns:")
print(clean.columns.to_list())
