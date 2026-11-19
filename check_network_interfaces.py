
from scapy.all import get_if_list, conf

print("Default interface:")
print(conf.iface)

print("\nAvailable network interfaces:")
for interface in get_if_list():
    print(interface)
