
import pandas as pd

df = pd.read_csv(
    "IoT Network Intrusion Dataset.csv",
    low_memory=False
)

arp_like = df[df["Protocol"] == 0]

print("Protocol 0 records:", len(arp_like))

print("\nCategory distribution:")
print(arp_like["Cat"].value_counts().to_string())

print("\nProtocol 0 feature statistics:")
columns = [
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
    "Flow_IAT_Max",
]

print(arp_like[columns].describe().T.to_string())

print("\nProtocol 6 and 17 category distribution:")
ip_traffic = df[df["Protocol"].isin([6, 17])]
print(pd.crosstab(
    ip_traffic["Protocol"],
    ip_traffic["Cat"]
).to_string())

print("\nProtocol 0 sample records:")
print(
    arp_like[
        ["Protocol", "Flow_ID", "Src_IP", "Dst_IP",
         "Src_Port", "Dst_Port", "Cat", "Sub_Cat"]
    ].head(15).to_string(index=False)
)

print("\nInspection complete. Original dataset unchanged.")
