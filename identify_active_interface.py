
from scapy.all import get_if_list, get_if_addr, conf

print("Checking network interfaces...\n")

for interface in get_if_list():
    try:
        ip_address = get_if_addr(interface)
        marker = " <-- DEFAULT" if interface == conf.iface else ""

        print(
            f"Interface: {interface}\n"
            f"IPv4: {ip_address}{marker}\n"
        )
    except Exception as error:
        print(f"Interface: {interface}")
        print(f"Could not read IPv4: {error}\n")
