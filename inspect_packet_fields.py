
from scapy.all import sniff, IP, TCP, UDP, ARP
from datetime import datetime

packet_count = 0

def inspect_packet(packet):
    global packet_count
    packet_count += 1

    print(f"\n--- Packet {packet_count} ---")
    print("Timestamp:", datetime.fromtimestamp(
        float(packet.time)
    ).strftime("%H:%M:%S.%f"))
    print("Packet length:", len(packet), "bytes")

    # Handle ARP separately because it is not an IP packet
    if packet.haslayer(ARP):
        arp = packet[ARP]
        print("Traffic type: ARP")
        print("Source IP:", arp.psrc)
        print("Destination IP:", arp.pdst)
        print("ARP operation:", arp.op)
        return

    if not packet.haslayer(IP):
        print("Traffic type: Non-IP or unsupported packet")
        return

    ip = packet[IP]
    print("Traffic type: IPv4")
    print("Source IP:", ip.src)
    print("Destination IP:", ip.dst)
    print("IP protocol number:", ip.proto)

    if packet.haslayer(TCP):
        tcp = packet[TCP]
        print("Transport: TCP")
        print("Source port:", tcp.sport)
        print("Destination port:", tcp.dport)
        print("TCP flags:", str(tcp.flags))

    elif packet.haslayer(UDP):
        udp = packet[UDP]
        print("Transport: UDP")
        print("Source port:", udp.sport)
        print("Destination port:", udp.dport)

    else:
        print("Transport: Other IP protocol")


print("Inspecting up to 10 packets for 20 seconds...")
print("Generate normal traffic by browsing a website.\n")

try:
    sniff(
        timeout=20,
        count=10,
        store=False,
        prn=inspect_packet
    )
    print(f"\nInspection finished. Packets inspected: {packet_count}")

except PermissionError:
    print("Permission denied. Try running VS Code as Administrator.")

except Exception as error:
    print("Capture error:", error)
