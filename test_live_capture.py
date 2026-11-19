
from scapy.all import sniff
from datetime import datetime

packet_count = 0

def show_packet(packet):
    global packet_count
    packet_count += 1

    print(f"\nPacket {packet_count}")
    print("Time:", datetime.now().strftime("%H:%M:%S"))
    print("Details:", packet.summary())

print("Starting live packet capture...")
print("Duration: 20 seconds")
print("Open a website on your own computer if traffic is low.")
print("Press Ctrl+C to stop early.\n")

try:
    sniff(
        timeout=20,
        count=10,
        store=False,
        prn=show_packet
    )

    print("\nCapture finished.")
    print("Packets observed:", packet_count)

except PermissionError:
    print("\nPermission denied.")
    print("Try running VS Code as Administrator.")

except Exception as error:
    print("\nCapture error:", error)
    print("Check Npcap and your network interface settings.")
