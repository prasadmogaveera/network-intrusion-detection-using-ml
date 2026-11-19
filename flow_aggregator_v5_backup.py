
from scapy.all import sniff, conf, get_if_addr, IP, TCP, UDP, ARP
from statistics import mean, pstdev

TARGET_IP = "192.168.1.3"
CAPTURE_SECONDS = 30
IDLE_TIMEOUT = 60

flows = {}
completed_flows = []
arp_count = 0
other_count = 0


def find_interface():
    for iface in conf.ifaces.values():
        try:
            if get_if_addr(iface) == TARGET_IP:
                return iface
        except Exception:
            continue
    return None


def make_key(protocol, src, dst):
    return protocol, tuple(sorted((src, dst)))


def new_flow(protocol, src, dst, timestamp):
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
        },
        "fin_directions": set()
    }


def finish_flow(key):
    flow = flows.pop(key, None)
    if flow is not None:
        completed_flows.append(flow)


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
    key = make_key(protocol, src, dst)
    timestamp = float(packet.time)

    # Expire inactive flows.
    expired = [
        k for k, f in flows.items()
        if timestamp - f["last_seen"] > IDLE_TIMEOUT
    ]
    for old_key in expired:
        finish_flow(old_key)

    if key not in flows:
        flows[key] = new_flow(protocol, src, dst, timestamp)

    flow = flows[key]
    flow["end"] = max(flow["end"], timestamp)
    flow["last_seen"] = timestamp

    packet_length = len(ip)
    flow["lengths"].append(packet_length)
    flow["times"].append(timestamp)

    if src == flow["first_src"] and dst == flow["first_dst"]:
        direction = "forward"
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += packet_length
    else:
        direction = "backward"
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += packet_length

    if protocol == 6:
        flags = int(transport.flags)

        for bit, name in [
            (0x02, "SYN"),
            (0x10, "ACK"),
            (0x04, "RST"),
            (0x01, "FIN"),
            (0x08, "PSH")
        ]:
            if flags & bit:
                flow["flags"][name] += 1

        if flags & 0x04:  # RST
            finish_flow(key)
            return

        if flags & 0x01:  # FIN
            flow["fin_directions"].add(direction)
            if len(flow["fin_directions"]) == 2:
                finish_flow(key)


def summarize(flow):
    lengths = flow["lengths"]
    times = sorted(flow["times"])
    iats = [
        (times[i] - times[i - 1]) * 1_000_000
        for i in range(1, len(times))
    ]

    return {
        "source": flow["first_src"],
         "destination": flow["first_dst"],
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
    print("Active interface not found. Check your connection.")
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

# Flows still active at the end of capture are finalized here.
for key in list(flows):
    finish_flow(key)

print("\nFLOW SUMMARY")
print("=" * 40)
print("Completed flows:", len(completed_flows))
print("ARP packets:", arp_count)
print("Other packets skipped:", other_count)

for i, flow in enumerate(completed_flows[:10], start=1):
    print(f"\nFlow {i}")
    for name, value in summarize(flow).items():
        print(f"  {name}: {round(value, 2) if isinstance(value, float) else value}")

print("\nTest complete. No dataset or packet capture was saved.")
