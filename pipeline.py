import numpy as np
import torch
import threading
from pylsl import StreamInlet, resolve_stream
import argparse
import time
from queue import Queue
from socket import *

TEST = True

class EEGThread(threading.Thread):
    def __init__(self, tID, winsize, stride):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.window = []
        self.winsize = winsize
        self.freq = 256
        self.stride = stride


    def run(self):
        print("EEG thread. Starting at {}".format(time.time()))
        strideCnt = 0
        try:
            if not TEST:
                streams = resolve_stream('type', 'EEG')
                inlet = StreamInlet(streams[0])
            while True:
                if TEST:
                    sample = [0.1, 0.2, 0.3, 0.4]
                    time.sleep(1/256)
                else:
                    sample, _ = inlet.pull_sample()
                # list, float
                if len(self.window) == self.winsize * self.freq:
                    self.window.pop(0)
                    self.window.append(sample[:4])
                    strideCnt += 1
                    if strideCnt == self.stride * self.freq:
                        # put window to buffer
                        dataBuffer.put(self.window.copy())
                        print('data produced!')
                        strideCnt = 0
                else:
                    self.window.append(sample[:4])
        except KeyboardInterrupt as e:
            print("Ending program", e)
        except Exception as e:
            print(e)
        finally:
            print("EEG thread. Exiting at {}".format(time.time()))

# NOT IN USE!
class InferenceThread(threading.Thread):
    def __init__(self, tID, model):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.model = model
    
    def run(self):
        print("Inference thread. Starting at {}".format(time.time()))
        try:
            while True:
                window = dataBuffer.get()
                print('data consumed!')
                data = np.expand_dims(np.transpose(np.array(window)), 0)
                # here run model
                workload = self.model(data)
                if workload < 0.25:
                    cmdBuffer.put('incr')
                    print('cmd incr produced!')
                elif workload > 0.75:
                    cmdBuffer.put('decr')
                    print('cmd decr produced!')
                else:
                    cmdBuffer.put('keep')
                    print('cmd keep produced!')
        except KeyboardInterrupt as e:
            print("Ending program", e)
        except Exception as e:
            print(e)
        finally:
            print("Inference thread. Exiting at {}".format(time.time()))


class MitigationThread(threading.Thread):
    def __init__(self, tID, modality):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.modality = modality
        self.state = 0

    def run(self):
        print("Mitigation thread. Starting at {}".format(time.time()))
        try:
            while True:
                cmd = cmdBuffer.get()
                print('cmd ' + cmd + ' consumed!')
                if cmd == 'incr':
                    pass # TODO: mitigation method
                elif cmd == 'decr':
                    pass
                else:
                    pass
        except KeyboardInterrupt as e:
            print("Ending program", e)
        except Exception as e:
            print(e)
        finally:
            print("Mitigation thread. Exiting at {}".format(time.time()))


def send_from(arr, dest):
    view = memoryview(arr).cast('B')
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]


def recv_into(arr, source):
    view = memoryview(arr).cast('B')
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

parser = argparse.ArgumentParser(description='Input for EEG AugCog system')
parser.add_argument('-winsize', type=int, default=30, help='window size (s)')
parser.add_argument('-stride', type=float, default=0.1, help='window size (s)')
parser.add_argument('-ip', type=str, default='localhost', help='ip address of the server')

opt = parser.parse_args()

dataBuffer = Queue()
cmdBuffer = Queue()

th1 = EEGThread(0, opt.winsize, opt.stride)
th2 = MitigationThread(1, '')
th1.start()
th2.start()

IP = opt.ip
PORT = 25000

while True:
    try:
        client = socket(AF_INET, SOCK_STREAM)
        client.connect((IP, PORT))
        print('Connected!')
        break
    except Exception as e:
        print(e)
        time.sleep(0.5)
        continue

while True:
    window = dataBuffer.get()  # block
    print('data consumed!')
    data = np.array(window, dtype=float) # l x c

    # send_from(data, client)
    # workload = int(str(client.recv(1024), 'utf-8'))
    # print('received! ', workload)
    # if workload < 0.25:
    #     cmdBuffer.put('incr')
    #     print('cmd incr produced!')
    # elif workload > 0.75:
    #     cmdBuffer.put('decr')
    #     print('cmd decr produced!')
    # else:
    #     cmdBuffer.put('keep')
    #     print('cmd keep produced!')

    try:
        print('here send data!')
        send_from(data, client)
        print('here data sent!')
        workload = int(str(client.recv(1024), 'utf-8'))
        print('received! ', workload)
        if workload < 0.25:
            cmdBuffer.put('incr')
            print('cmd incr produced!')
        elif workload > 0.75:
            cmdBuffer.put('decr')
            print('cmd decr produced!')
        else:
            cmdBuffer.put('keep')
            print('cmd keep produced!')
    # except KeyboardInterrupt as e:
    #     client.close()
    except Exception as e:
        print(e)
        try:
            client.close()
            client = socket(AF_INET, SOCK_STREAM)
            client.connect(('localhost', 25000))
        except Exception as e:
            print('try again:', e)






