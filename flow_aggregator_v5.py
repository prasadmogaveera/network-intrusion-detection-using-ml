
import os

from scapy.all import sniff, conf, get_if_addr, IP, TCP, UDP, ARP
from statistics import mean, pstdev


CAPTURE_SECONDS = 30
# The training data contains flow durations below 100,000 microseconds.
# Keep live records within the same capture-window scale.
FLOW_WINDOW_SECONDS = 0.1
IDLE_TIMEOUT = FLOW_WINDOW_SECONDS

flows = {}
completed_flows = []
arp_count = 0
other_count = 0


def find_interface():
    """Find the active capture interface from the default route."""
    configured_ip = os.getenv("CAPTURE_IP", "").strip()
    if configured_ip:
        for iface in conf.ifaces.values():
            try:
                if get_if_addr(iface) == configured_ip:
                    return iface
            except (AttributeError, OSError, RuntimeError):
                continue
        return None

    try:
        interface, source_ip, _ = conf.route.route("0.0.0.0")
        if interface and source_ip != "0.0.0.0":
            return interface
    except (AttributeError, OSError, RuntimeError, ValueError):
        pass

    for iface in conf.ifaces.values():
        try:
            address = get_if_addr(iface)
            if address not in {"0.0.0.0", "127.0.0.1"}:
                return iface
        except (AttributeError, OSError, RuntimeError):
            continue
    return None


def make_key(protocol, src, dst):
    """Create the same key for packets travelling in either direction."""
    return protocol, tuple(sorted((src, dst)))


def new_flow(protocol, src, dst, timestamp):
    """Initialize a new bidirectional flow."""
    return {
        "protocol": protocol,
        "first_src_port": src[1],
        "first_dst_port": dst[1],
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
            "SYN": 0,
            "ACK": 0,
            "RST": 0,
            "FIN": 0,
            "PSH": 0,
        },
        "fin_directions": set(),
    }


def finish_flow(key):
    """Move a completed flow from active to completed flows."""
    flow = flows.pop(key, None)

    if flow is not None and flow["lengths"]:
        completed_flows.append(flow)


def process_packet(packet):
    """Process one captured packet and update its flow."""
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

    # Finish flows that have been inactive for too long.
    expired = [
        flow_key
        for flow_key, flow in list(flows.items())
        if timestamp - flow["last_seen"] > IDLE_TIMEOUT
    ]

    for flow_key in expired:
        finish_flow(flow_key)

    # Start a new flow if this connection is not active.
    if (
        key in flows
        and timestamp - flows[key]["start"] >= FLOW_WINDOW_SECONDS
    ):
        finish_flow(key)

    if key not in flows:
        flows[key] = new_flow(protocol, src, dst, timestamp)

    flow = flows[key]

    flow["start"] = min(flow["start"], timestamp)
    flow["end"] = max(flow["end"], timestamp)
    flow["last_seen"] = max(flow["last_seen"], timestamp)

    # This measures IP packet length, excluding the Ethernet header.
    packet_length = len(ip)

    flow["lengths"].append(packet_length)
    flow["times"].append(timestamp)

    # The first observed packet establishes the forward direction.
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
            (0x08, "PSH"),
        ]:
            if flags & bit:
                # In the source dataset these selected flag fields are
                # binary indicators, not counts of flagged packets.
                flow["flags"][name] = 1

        # A reset closes the TCP flow.
        if flags & 0x04:
            finish_flow(key)
            return

        # Close after FIN packets have been observed in both directions.
        if flags & 0x01:
            flow["fin_directions"].add(direction)

            if len(flow["fin_directions"]) == 2:
                finish_flow(key)


def summarize(flow):
    """Calculate the features currently supported by this aggregator."""
    lengths = flow["lengths"]
    times = sorted(flow["times"])

    if not lengths:
        return None

    if len(lengths) != flow["fwd_packets"] + flow["bwd_packets"]:
        raise ValueError(
            "Flow packet invariant failed: stored packet lengths "
            f"({len(lengths)}) != directional packet total "
            f"({flow['fwd_packets'] + flow['bwd_packets']})."
        )

    if sum(lengths) != flow["fwd_bytes"] + flow["bwd_bytes"]:
        raise ValueError(
            "Flow byte invariant failed: packet-length sum "
            f"({sum(lengths)}) != directional byte total "
            f"({flow['fwd_bytes'] + flow['bwd_bytes']})."
        )

    # Inter-arrival times between consecutive captured packets.
    iats = [
        (times[i] - times[i - 1]) * 1_000_000
        for i in range(1, len(times))
    ]

    return {
        "source": flow["first_src"],
        "destination": flow["first_dst"],
        "Src_Port": flow["first_src_port"],
        "Dst_Port": flow["first_dst_port"],
        "protocol": flow["protocol"],
        "duration_us": (
            max(times) - min(times)
        ) * 1_000_000,
        "total_packets": (
            flow["fwd_packets"] + flow["bwd_packets"]
        ),
        "total_bytes": (
            flow["fwd_bytes"] + flow["bwd_bytes"]
        ),
        "fwd_packets": flow["fwd_packets"],
        "bwd_packets": flow["bwd_packets"],
        "fwd_bytes": flow["fwd_bytes"],
        "bwd_bytes": flow["bwd_bytes"],
        "packet_length_min": min(lengths),
        "packet_length_max": max(lengths),
        "packet_length_mean": mean(lengths),
        "packet_length_std": pstdev(lengths),
        "mean_iat_us": mean(iats) if iats else 0.0,
        "std_iat_us": pstdev(iats) if iats else 0.0,
        **flow["flags"],
    }


def capture_traffic():
    """Capture traffic and print a short flow summary."""
    global arp_count, other_count

    interface = find_interface()

    if interface is None:
        print("Active interface not found. Check your connection.")
        return

    # Reset previous results when starting a fresh capture.
    flows.clear()
    completed_flows.clear()
    arp_count = 0
    other_count = 0

    print("Interface:", interface)
    print(f"Capturing for {CAPTURE_SECONDS} seconds...")
    print("Browse a few websites while capture runs.\n")

    try:
        sniff(
            iface=interface,
            timeout=CAPTURE_SECONDS,
            store=False,
            prn=process_packet,
        )
    except PermissionError:
        print("Permission denied. Check capture permissions.")
        return
    except Exception as error:
        print("Capture error:", error)
        return

    # Finalize flows still active when capture ends.
    for key in list(flows):
        finish_flow(key)

    print("\nFLOW SUMMARY")
    print("=" * 40)
    print("Completed flows:", len(completed_flows))
    print("ARP packets:", arp_count)
    print("Other packets skipped:", other_count)

    for i, flow in enumerate(completed_flows[:10], start=1):
        print(f"\nFlow {i}")

        result = summarize(flow)
        if result is None:
            continue

        for name, value in result.items():
            if isinstance(value, float):
                value = round(value, 2)
            print(f"  {name}: {value}")

    print("\nCapture complete. No traffic was blocked or saved.")


def main():
    capture_traffic()


# Important: importing this module must NOT start a capture.
if __name__ == "__main__":
    main()
