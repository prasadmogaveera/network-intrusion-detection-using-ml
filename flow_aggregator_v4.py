
from collections import defaultdict
from statistics import mean, pstdev
from scapy.all import sniff, conf, get_if_addr, IP, TCP, UDP, ARP

TARGET_IP = "192.168.1.3"
CAPTURE_SECONDS = 30
IDLE_TIMEOUT = 60

flows = {}
arp_count = 0
other_count = 0
completed_flows = []


def find_interface():
    for iface in conf.ifaces.values():
        try:
            if get_if_addr(iface) == TARGET_IP:
                return iface
        except Exception:
            pass
    return None


def flow_key(protocol, src, dst):
    return (protocol, tuple(sorted((src, dst))))


def create_flow(protocol, src, dst, timestamp):
    return {
        "protocol": protocol,
        "first_src": src,
        "first_dst": dst,
        "start": timestamp,
        "end": timestamp,
        "last_seen": timestamp,
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
    key = flow_key(protocol, src, dst)
    timestamp = float(packet.time)
    packet_length = len(ip)

    # Expire old flows before processing this packet.
    expired = [
        k for k, f in flows.items()
        if timestamp - f["last_seen"] > IDLE_TIMEOUT
    ]
    for old_key in expired:
        completed_flows.append(flows.pop(old_key))

    # Start a new flow if none exists or the previous one expired.
    if key not in flows:
        flows[key] = create_flow(protocol, src, dst, timestamp)

    flow = flows[key]
    flow["end"] = max(flow["end"], timestamp)
    flow["last_seen"] = timestamp
    flow["lengths"].append(packet_length)
    flow["times"].append(timestamp)

    if src == flow["first_src"] and dst == flow["first_dst"]:
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += packet_length
    else:
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += packet_length

    if protocol == 6:
        flags = int(transport.flags)
        if flags & 0x02:
            flow["flags"]["SYN"] += 1
        if flags & 0x10:
            flow["flags"]["ACK"] += 1
        if flags & 0x04:
            flow["flags"]["RST"] += 1
        if flags & 0x01:
            flow["flags"]["FIN"] += 1
        if flags & 0x08:
            flow["flags"]["PSH"] += 1

        # A TCP connection may end on FIN or RST.
        if flags & (0x01 | 0x04):
            completed_flows.append(flows.pop(key))


def summarize_flow(flow):
    lengths = flow["lengths"]
    times = sorted(flow["times"])
    iats = [
        (times[i] - times[i - 1]) * 1_000_000
        for i in range(1, len(times))
    ]

    return {
        "protocol": flow["protocol"],
        "duration_us": (flow["end"] - flow["start"]) * 1_000_000,
        "total_packets": flow["fwd_packets"] + flow["bwd_packets"],
        "total_bytes": flow["fwd_bytes"] + flow["bwd_bytes"],
        "fwd_packets": flow["fwd_packets"],
        "bwd_packets": flow["bwd_packets"],
        "fwd_bytes": flow["fwd_bytes"],
        "bwd_bytes": flow["bwd_bytes"],
        "packet_length_min": min(lengths),
        "packet_length_max": max(lengths),
        "packet_length_mean": mean(lengths),
        "packet_length_std": pstdev(lengths),
        "mean_iat_us": mean(iats) if iats else 0,
        **flow["flags"]
    }


interface = find_interface()

if interface is None:
    print("Active interface not found. Check your network connection.")
    raise SystemExit

print("Interface:", interface)
print(f"Capturing for {CAPTURE_SECONDS} seconds...")
print("Browse a few websites while capture runs.\n")

try:
    sniff(
        iface=interface,
        timeout=CAPTURE_SECONDS,
        store=False,
        prn=process_packet
    )
except Exception as error:
    print("Capture error:", error)
    raise SystemExit

# Finalize remaining flows at the end of this capture.
completed_flows.extend(flows.values())
flows.clear()

print("\nFLOW AGGREGATION SUMMARY")
print("=" * 40)
print("Completed flows:", len(completed_flows))
print("ARP packets:", arp_count)
print("Other packets skipped:", other_count)

for index, flow in enumerate(completed_flows[:10], start=1):
    features = summarize_flow(flow)
    print(f"\nFlow {index}:")
    for name, value in features.items():
        if isinstance(value, float):
            value = round(value, 2)
        print(f"  {name}: {value}")

print("\nTest complete. No dataset or packet capture was saved.")
