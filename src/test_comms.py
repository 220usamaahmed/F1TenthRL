import socket
import pickle
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import signal

obs_data = None
angle_max, angle_min = None, None
should_exit = False
TICK_RATE = 0.1


def send_full_message(sock, obj):
    message = pickle.dumps(obj)
    msg_length = len(message)
    sock.sendall(msg_length.to_bytes(4, "big"))
    sock.sendall(message)


def receive_full_message(conn):
    msg_length_data = conn.recv(4)
    if not msg_length_data:
        return None

    msg_length = int.from_bytes(msg_length_data, "big")

    data_chunks = []
    bytes_received = 0
    while bytes_received < msg_length:
        chunk = conn.recv(min(1024, msg_length - bytes_received))
        if not chunk:
            break

        data_chunks.append(chunk)
        bytes_received += len(chunk)

    if bytes_received < msg_length:
        return None

    return pickle.loads(b"".join(data_chunks))


def plot_lidar_data(scan_lines, angle_min, angle_max):
    angles = np.linspace(angle_min, angle_max, len(scan_lines), endpoint=False)
    distances = np.array(scan_lines)
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return x, y


def update_plot(frame, plot):
    global obs_data
    if obs_data is None:
        return (plot,)

    scan_lines = obs_data["scan"]["scan_lines"]
    angle_min = obs_data["scan"]["angle_min"]
    angle_max = obs_data["scan"]["angle_max"]
    x, y = plot_lidar_data(scan_lines, angle_min, angle_max)

    plot.set_data(x, y)
    return (plot,)


def receive_data(client):
    global obs_data, should_exit

    try:
        while not should_exit:
            obs_data = receive_full_message(client)
            if not obs_data:
                print("Server disconnected")
                break
            print(f"Received: {obs_data.keys()}")
            print(f"Scans: {len(obs_data['scan']['scan_lines'])}")
            send_full_message(client, {"speed": 1.0, "steering_angle": 0.0})

        send_full_message(client, {"speed": 0.0, "steering_angle": 0.0})
    except KeyboardInterrupt:
        print("Data thread interrupted")
    finally:
        client.close()


def signal_handler(sig, frame):
    global should_exit
    print("Exiting...")
    should_exit = True


def main():
    global obs_data

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("localhost", 3000))
    client.setblocking(True)

    signal.signal(signal.SIGINT, signal_handler)

    fig, ax = plt.subplots()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    (plot,) = ax.plot([], [], "bo", markersize=2)

    ani = FuncAnimation(fig, update_plot, fargs=(plot,), interval=1 / TICK_RATE)

    data_thread = threading.Thread(target=receive_data, args=(client,))
    data_thread.daemon = True
    data_thread.start()

    plt.show()


if __name__ == "__main__":
    main()
