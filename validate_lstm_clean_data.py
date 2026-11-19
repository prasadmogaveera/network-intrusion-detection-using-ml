
import pandas as pd

CLEAN_FILE = "IoT_Network_Intrusion_Clean_Binary.csv"
LSTM_FILE = "IoT_Network_Intrusion_LSTM_Clean.csv"

FEATURES = [
    "Protocol", "Src_Port", "Dst_Port", "Flow_Duration",
    "Tot_Fwd_Pkts", "Tot_Bwd_Pkts",
    "TotLen_Fwd_Pkts", "TotLen_Bwd_Pkts",
    "Pkt_Len_Min", "Pkt_Len_Max", "Pkt_Len_Mean",
    "Pkt_Len_Std", "Flow_IAT_Mean", "Flow_IAT_Std",
    "SYN_Flag_Cnt", "ACK_Flag_Cnt", "RST_Flag_Cnt",
    "FIN_Flag_Cnt", "PSH_Flag_Cnt"
]

clean = pd.read_csv(CLEAN_FILE)
lstm = pd.read_csv(LSTM_FILE)

print("CLEAN BINARY DATASET")
print("Rows:", len(clean))
print("Unique feature-label combinations:",
      len(clean.drop_duplicates(subset=FEATURES + ["Label"])))

print("\nLSTM DATASET")
print("Rows:", len(lstm))
print("Unique feature-label combinations:",
      len(lstm.drop_duplicates(subset=FEATURES + ["Label"])))

print("\nDuplicate feature-label rows in LSTM dataset:",
      lstm.duplicated(subset=FEATURES + ["Label"]).sum())

print("\nMissing timestamps:", lstm["Timestamp"].isna().sum())

print("\nLabels in clean dataset:")
print(clean["Label"].value_counts())

print("\nLabels in LSTM dataset:")
print(lstm["Label"].value_counts())
