
from scapy.all import sniff, conf, get_if_addr

# Use the interface associated with your active local IPv4 address.
TARGET_IP = "192.168.1.3"

interface = None

for iface in conf.ifaces.values():
    try:
        if get_if_addr(iface) == TARGET_IP:
            interface = iface
            break
    except Exception:
        pass

if interface is None:
    print("Could not find the active interface.")
    print("Check your network connection and run the interface check again.")
    raise SystemExit

print("Selected interface:", interface)
print("Local IPv4:", TARGET_IP)
print("\nCapturing packets for 20 seconds...")
print("Browse a few websites while the capture runs.\n")

packet_count = 0
tcp_count = 0
udp_count = 0
arp_count = 0
other_count = 0


def inspect_packet(packet):
    global packet_count, tcp_count, udp_count, arp_count, other_count

    packet_count += 1

    from scapy.all import IP, TCP, UDP, ARP

    if packet.haslayer(ARP):
        arp_count += 1
    elif packet.haslayer(IP) and packet.haslayer(TCP):
        tcp_count += 1
    elif packet.haslayer(IP) and packet.haslayer(UDP):
        udp_count += 1
    else:
        other_count += 1


try:
    sniff(
        iface=interface,
        timeout=20,
        store=False,
        prn=inspect_packet
    )
except PermissionError:
    print("Permission denied. Check Npcap permissions.")
    raise SystemExit
except Exception as error:
    print("Capture error:", error)
    raise SystemExit

print("CAPTURE SUMMARY")
print("------------------------------")
print("Total packets:", packet_count)
print("TCP packets:", tcp_count)
print("UDP packets:", udp_count)
print("ARP packets:", arp_count)
print("Other packets:", other_count)
print("\nTest complete. No packets were saved.")
