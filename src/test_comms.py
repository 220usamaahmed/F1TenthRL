import socket


def send_full_message(sock, message):
    # Send message length first (as 4-byte integer)
    msg_length = len(message)
    sock.sendall(msg_length.to_bytes(4, "big"))  # Convert int to bytes and send

    # Send the actual message
    sock.sendall(message)


def receive_full_message(conn):
    # First, receive the message length (assume it's sent as a fixed 4-byte integer)
    msg_length_data = conn.recv(4)
    if not msg_length_data:
        return None  # Client disconnected

    msg_length = int.from_bytes(msg_length_data, "big")  # Convert bytes to int

    # Now receive the full message in chunks
    data_chunks = []
    bytes_received = 0
    while bytes_received < msg_length:
        chunk = conn.recv(
            min(1024, msg_length - bytes_received)
        )  # Receive up to 1024 bytes
        if not chunk:
            break  # Client disconnected unexpectedly

        data_chunks.append(chunk)
        bytes_received += len(chunk)

    return b"".join(data_chunks)  # Combine chunks into the full message


client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(("localhost", 3000))

try:
    while True:
        data = receive_full_message(client)
        if not data:
            print("Server disconnected")
            break
        print(f"Received: {data.decode()}")
except KeyboardInterrupt:
    client.close()


# try:
#     while True:
#         message = input("Enter message (or 'exit' to quit): ")
#         if message.lower() == "exit":
#             break

#         send_full_message(client, message.encode())  # Send message properly
#         response = client.recv(1024)
#         print(f"Server replied: {response.decode()}")
# except KeyboardInterrupt:
#     ...
# finally:
#     client.close()
