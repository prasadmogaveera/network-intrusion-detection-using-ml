
from statistics import mean, pstdev
from scapy.all import sniff, IP, TCP, UDP, ARP

flows = {}
arp_count = 0
other_count = 0

FLAG_MAP = {
    "S": "SYN",
    "A": "ACK",
    "R": "RST",
    "F": "FIN",
    "P": "PSH",
}


def process_packet(packet):
    global arp_count, other_count

    if packet.haslayer(ARP):
        arp_count += 1
        return

    if not packet.haslayer(IP):
        other_count += 1
        return

    ip = packet[IP]

    if packet.haslayer(TCP):
        transport = packet[TCP]
        protocol = 6
    elif packet.haslayer(UDP):
        transport = packet[UDP]
        protocol = 17
    else:
        other_count += 1
        return

    src = (ip.src, int(transport.sport))
    dst = (ip.dst, int(transport.dport))
    key = (protocol, tuple(sorted((src, dst))))

    timestamp = float(packet.time)
    packet_length = len(ip)

    if key not in flows:
        flows[key] = {
            "protocol": protocol,
            "first_src": src,
            "first_dst": dst,
            "start": timestamp,
            "end": timestamp,
            "fwd_packets": 0,
            "bwd_packets": 0,
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "lengths": [],
            "times": [],
            "flags": {
                "SYN": 0, "ACK": 0, "RST": 0,
                "FIN": 0, "PSH": 0
            }
        }

    flow = flows[key]
    flow["end"] = max(flow["end"], timestamp)
    flow["lengths"].append(packet_length)
    flow["times"].append(timestamp)

    if src == flow["first_src"] and dst == flow["first_dst"]:
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += packet_length
    else:
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += packet_length

    if protocol == 6:
        flags = str(transport.flags)
        for flag_letter, flag_name in FLAG_MAP.items():
            if flag_letter in flags:
                flow["flags"][flag_name] += 1


print("Capturing normal TCP/UDP traffic for 20 seconds...")
print("Browse a few websites while capture runs.\n")

try:
    sniff(timeout=20, store=False, prn=process_packet)
except PermissionError:
    print("Permission denied. Try running VS Code as Administrator.")
    raise SystemExit
except Exception as error:
    print("Capture error:", error)
    raise SystemExit

print("\nFLOW SUMMARY")
print("=" * 45)
print("Flow candidates:", len(flows))
print("ARP packets:", arp_count)
print("Other packets skipped:", other_count)

for i, flow in enumerate(flows.values(), start=1):
    lengths = flow["lengths"]
    times = sorted(flow["times"])
    iats = [
        (times[j] - times[j - 1]) * 1_000_000
        for j in range(1, len(times))
    ]

    print(f"\nFlow {i}: protocol={flow['protocol']}")
    print("Packets forward/backward:",
          flow["fwd_packets"], "/", flow["bwd_packets"])
    print("Bytes forward/backward:",
          flow["fwd_bytes"], "/", flow["bwd_bytes"])
    print("Duration (microseconds):",
          round((flow["end"] - flow["start"]) * 1_000_000, 2))
    print("Packet length min/max:",
          min(lengths), "/", max(lengths))
    print("Packet length mean/std:",
          round(mean(lengths), 2), "/", round(pstdev(lengths), 2))
    print("Mean IAT (microseconds):",
          round(mean(iats), 2) if iats else 0)
    print("TCP flags:", flow["flags"])

print("\nTest complete. No dataset or packet capture was saved.")
