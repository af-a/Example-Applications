#!/usr/bin/env python3

"""
Listens to a UDP port and prints any received messages.
"""

import time
import signal
import socket
import json

if __name__ == '__main__':
    udp_ip = '10.157.174.66'
    # Note: Raises: " PermissionError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions"
    # udp_port = 5000
    udp_port = 7000
    buffer_size = 1024

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((udp_ip, udp_port))

    print(f'[udp_receiver] Listening for UDP messages over IP ' + \
                  f' {udp_ip} and port {udp_port}...')

    try:
        while True:
            signal.signal(signal.SIGINT, signal.SIG_DFL);
            data, addr = sock.recvfrom(buffer_size)

            decoded_data_json_dict = json.loads(data.decode())
            print(f'[DEBUG] decoded_data_json_dict: {decoded_data_json_dict}')

            print(f'[DEBUG] addr: {addr}')
            # print(f'[DEBUG] sys.getsizeof(data): {sys.getsizeof(data)}')

            print(f'[INFO] Received message: \n{decoded_data_json_dict}')
            print(f'[INFO] Time to receive sent message: {time.time() - decoded_data_json_dict['timestamp']}')

            time.sleep(0.01)
    except (KeyboardInterrupt):
        print('[udp_receiver] Stopping receiver')
