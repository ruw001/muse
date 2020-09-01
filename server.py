from socket import *
import numpy as np
import argparse
import time


def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]


def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    timeout = 20
    ts_start = 0
    started = False
    total = 0
    while len(view):
        nrecv = source.recv_into(view)
        if nrecv == 0:
            if not started:
                started = True
                ts_start = time.time()
            else:
                total = time.time() - ts_start
                if total >= timeout:
                    return 1
        else:
            started = False
            total = 0
        view = view[nrecv:]
    return 0

parser = argparse.ArgumentParser(description='Input for EEG AugCog system')
parser.add_argument('-winsize', type=int, default=30, help='window size (s)')

opt = parser.parse_args()
HOST = ''
PORT = 25000
freq = 256
window = np.zeros((opt.winsize * freq, 4))

with socket(AF_INET, SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    count = 0
    server.listen()
    conn, addr = server.accept()
    while True:
        try:
            status = recv_into(window, conn)
            if status == 1:
                print('timeout!')
                conn.close()
                conn, addr = server.accept()
                print('accepted!')
                continue
            print('data received!' + str(count))
            print(window)
            print('predicting...')
            time.sleep(0.5)
            conn.send(bytes(str(count), 'utf-8'))
            print('result sent!' + str(count))
            count += 1
        except Exception as e:
            print(e)

