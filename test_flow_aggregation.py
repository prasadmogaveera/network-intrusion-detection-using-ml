
from collections import defaultdict
from statistics import mean, pstdev
from scapy.all import sniff, IP, TCP, UDP, ARP

flows = {}
arp_count = 0
other_count = 0


def get_endpoint(ip, port):
    return (ip, int(port))


def process_packet(packet):
    global arp_count, other_count

    timestamp = float(packet.time)

    # ARP is not an IP flow. Count it separately for now.
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

    src = get_endpoint(ip.src, transport.sport)
    dst = get_endpoint(ip.dst, transport.dport)

    # Canonical key groups both directions into the same flow.
    endpoints = tuple(sorted((src, dst)))
    key = (protocol, endpoints)

    if key not in flows:
        flows[key] = {
            "protocol": protocol,
            "first_src": src,
            "first_dst": dst,
            "start": timestamp,
            "end": timestamp,
            "packets": [],
            "fwd_packets": 0,
            "bwd_packets": 0,
            "fwd_bytes": 0,
            "bwd_bytes": 0,
            "flags": {
                "SYN": 0, "ACK": 0, "RST": 0,
                "FIN": 0, "PSH": 0
            },
            "times": []
        }

    flow = flows[key]
    flow["end"] = max(flow["end"], timestamp)

    # Use IP-layer length, not Ethernet frame length.
    packet_length = len(ip)

    flow["packets"].append(packet_length)
    flow["times"].append(timestamp)

    if src == flow["first_src"] and dst == flow["first_dst"]:
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += packet_length
    else:
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += packet_length

    if protocol == 6:
        flags = transport.flags
        for flag in flow["flags"]:
            if flag in str(flags):
                flow["flags"][flag] += 1


print("Capturing TCP/UDP flows for 20 seconds...")
print("Browse normally to generate traffic.\n")

try:
    sniff(timeout=20, store=False, prn=process_packet)

except PermissionError:
    print("Permission denied. Try running VS Code as Administrator.")
    raise SystemExit

except Exception as error:
    print("Capture error:", error)
    raise SystemExit


print("\nFLOW AGGREGATION RESULTS")
print("=" * 50)
print("Bidirectional TCP/UDP flows:", len(flows))
print("ARP packets observed:", arp_count)
print("Other packets skipped:", other_count)

for number, flow in enumerate(flows.values(), start=1):
    lengths = flow["packets"]
    times = sorted(flow["times"])
    iats = [
        (times[i] - times[i - 1]) * 1_000_000
        for i in range(1, len(times))
    ]

    print(f"\nFlow {number}")
    print("Protocol:", flow["protocol"])
    print("First observed direction:",
          flow["first_src"], "->", flow["first_dst"])
    print("Forward packets:", flow["fwd_packets"])
    print("Backward packets:", flow["bwd_packets"])
    print("Forward bytes:", flow["fwd_bytes"])
    print("Backward bytes:", flow["bwd_bytes"])
    print("Duration (microseconds):",
          round((flow["end"] - flow["start"]) * 1_000_000, 2))
    print("Packet length min/max:",
          min(lengths), max(lengths))
    print("Packet length mean:",
          round(mean(lengths), 2))
    print("Packet length population std:",
          round(pstdev(lengths), 2))
    print("Mean inter-arrival time (microseconds):",
          round(mean(iats), 2) if iats else 0)
    print("TCP flags:", flow["flags"])

print("\nPrototype finished. No traffic or dataset was saved.")
