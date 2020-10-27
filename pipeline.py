import numpy as np
import torch
import threading
from pylsl import StreamInlet, resolve_stream
import argparse
import time
import tkinter
from queue import Queue
from socket import *
import random
import logging
import os

WAIT_FOR_KEY = False
BEST_REACT_TIME = None

class EEGThread(threading.Thread):
    def __init__(self, tID, winsize, stride):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.window = []
        self.winsize = winsize
        self.freq = 256
        self.stride = stride

    def run(self):
        print("EEG thread. Starting at {}, winsize: {}, stride: {}".format(time.time(), self.winsize, self.stride))
        strideCnt = 0
        try:
            if not TEST:
                streams = resolve_stream('type', 'EEG')
                inlet = StreamInlet(streams[0])
            while True:
                if TEST:
                    sample = [0.1, 0.2, 0.3, 0.4]
                    time.sleep(0.001)
                else:
                    sample, _ = inlet.pull_sample()
                # list, float
                if len(self.window) == self.winsize * self.freq:
                    self.window.pop(0)
                    self.window.append(sample[:4])
                    strideCnt += 1
                    # print('window full', strideCnt)
                    if strideCnt == self.stride * self.freq:
                        # put window to buffer
                        dataBuffer.put(self.window.copy())
                        print('data produced!')
                        strideCnt = 0
                else:
                    self.window.append(sample[:4])
                    # print('add to window...', len(self.window))
        except KeyboardInterrupt as e:
            print("Ending program", e)
        except Exception as e:
            print(e)
        finally:
            print("EEG thread. Exiting at {}".format(time.time()))

class InferenceThread(threading.Thread):
    def __init__(self, tID, IP, PORT):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.IP = IP
        self.PORT = PORT #25000
    
    def run(self):
        while True:
            try:
                client = socket(AF_INET, SOCK_STREAM)
                client.connect((self.IP, self.PORT))
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
            try:
                print('here send data!')
                send_from(data, client)
                print('here data sent!')
                workload = int(str(client.recv(1024), 'utf-8'))
                print('received! ', workload)
                cmdBuffer.put(workload)
            except KeyboardInterrupt as e:
                client.close()
            except Exception as e:
                print(e)
                try:
                    client.close()
                    client = socket(AF_INET, SOCK_STREAM)
                    client.connect((self.IP, self.PORT))
                except Exception as e:
                    print('try again:', e)

class MitigationThread(threading.Thread):
    def __init__(self, tID, avgSize):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.window = []
        self.averageSzie = avgSize
        self.STATE = 0
    
    def run(self):
        print("Mitigation thread. Starting at {}".format(time.time()))
        try:
            while True:
                cmd = cmdBuffer.get()
                if len(self.window) >= self.averageSzie:
                    self.window.pop(0)
                self.window.append(cmd)
                self.STATE = self.vote()
                print('STATE:', self.STATE)
        except KeyboardInterrupt as e:
            print("Ending program", e)
        except Exception as e:
            print(e)
        finally:
            print("Mitigation thread. Exiting at {}".format(time.time()))

    def vote(self):
        dic = {}
        for wl in self.window:
            if wl not in dic:
                dic[wl] = 0
            dic[wl] += 1
        vs = list(dic.values())
        ks = list(dic.keys())
        return ks[vs.index(max(vs))]

def N_back_dynamic(user_id, interval, length, tasks, random_=False):
    global WAIT_FOR_KEY, BEST_REACT_TIME
    logging.basicConfig(filename=os.path.join(user_id, 'task_{}.log'.format('Random' if random_ else 'Adaptive')), 
    level=logging.INFO)
    if random_:
        random.shuffle(tasks)
    else:
        tasks.sort()
    dash_int = 0.5

    logging.info('Task {}, length={}, start!'.format('Random' if random_ else 'Adaptive', length))
    react_time = {}
    correct_react = {}

    for t in tasks:
        if t not in react_time:
            react_time[t] = []
            correct_react[t] = 0

    next_task = 0
    while True:
        curr_task = tasks[next_task]
        seq, res = randomTaskGenerator(curr_task, length)
        title.config(text='{}-back'.format(curr_task))
        logging.info('Task {}-back start!'.format(curr_task))
        for i in range(length):
            WAIT_FOR_KEY = True
            start_timer = time.time()
            number.config(text=seq[i])
            time.sleep(interval - dash_int)
            WAIT_FOR_KEY = False
            if res[i] == 1:
                if BEST_REACT_TIME == None:
                    feedback.config(text='False', fg='red')
                    react_time[curr_task].append(interval - dash_int)
                    logging.info('{},{},{},{}'.format(seq[i], res[i], False, interval - dash_int))
                else:
                    feedback.config(text='True', fg='light green')
                    react_time[curr_task].append(BEST_REACT_TIME - start_timer)
                    logging.info('{},{},{},{}'.format(seq[i], res[i], True, BEST_REACT_TIME - start_timer))
                    correct_react[curr_task] += 1
            else:
                if BEST_REACT_TIME == None:
                    feedback.config(text='True', fg='light green')
                    logging.info('{},{},{},{}'.format(seq[i], res[i], True, 'N/A'))
                    correct_react[curr_task] += 1
                else:
                    feedback.config(text='False', fg='red')
                    logging.info('{},{},{},{}'.format(seq[i], res[i], False, 'N/A'))
            BEST_REACT_TIME = None
            number.config(text='-')
            time.sleep(dash_int)
            feedback.config(text='', fg='black')
        if random_:
            next_task += 1
            if next_task == len(tasks):
                title.config(text='Finish!')
                results = loggingResult(correct_react, react_time)
                for r in results:
                    logging.info(r)
                break
        else:
            wl = th3.STATE
            del tasks[next_task]
            if tasks == []:
                title.config(text='Finish!')
                results = loggingResult(correct_react, react_time)
                for r in results:
                    logging.info(r)
                break
            if wl == 0:
                tasks.sort(reverse=True)
            else:
                tasks.sort()
        title.config(text='Next task: {}-back...'.format(tasks[next_task]))
        time.sleep(5)

def loggingResult(corrects, times):
    results = []
    total_times = []
    total_corrects = 0
    for key in corrects:
        curr_times = times[key]
        curr_corrects = corrects[key]
        total_times += curr_times
        total_corrects += curr_corrects
        s = str(key) + '-back task: average react time: {}, correct react: {}'.format(sum(curr_times)/len(curr_times), curr_corrects)
        results.append(s)
    s = 'Total average react time: {}, correct react: {}'.format(sum(total_times)/len(total_times), total_corrects)
    results.append(s)
    return results
    


def check(event=None):
    global WAIT_FOR_KEY, BEST_REACT_TIME
    if WAIT_FOR_KEY and BEST_REACT_TIME == None:
        BEST_REACT_TIME = time.time()

def startTask(user_id, interval, length, tasks, random_=False):
    '''
    user_id::str        user id
    interval::float     between each number (2.25s by default)
    length::int         the length of the sequence
    tasks::[int]        list of tasks
    '''

    if not os.path.exists(user_id):
        os.mkdir(user_id)

    # start recording
    th_task = threading.Thread(target=N_back_dynamic, args=(user_id, interval, length, tasks, random_))

    th_task.daemon = True
    
    th_task.start()
    print('N-back thread is started!')

def randomTaskGenerator(N, length):
    '''
    make sure at least 10 matches in the sequence
    '''
    print('Generating sequence...')
    seq = [''] * length
    res = [0] * length
    letters = ['B', 'F', 'H', 'J', 'L', 'M', 'Q', 'R', 'X']
    for i in range(10):
        pairhead = random.randint(0, length-1-N)
        while seq[pairhead] != '' or seq[pairhead + N] != '':
            pairhead = random.randint(0, length-1-N)
        num = random.randint(0,8)
        seq[pairhead] = letters[num]
        seq[pairhead + N] = letters[num]
    for i in range(length):
        if seq[i] == '':
            num = random.randint(0,8)
            seq[i] = letters[num]
    for i in range(N, length):
        if seq[i-N] == seq[i]:
            res[i] = 1
    
    print('Sequence generated!')
    return seq, res
    

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
parser.add_argument('-winsize', type=int, default=45, help='window size (s)')
parser.add_argument('-stride', type=float, default=1, help='stride (s)')
parser.add_argument('-ip', type=str, default='137.110.115.9', help='ip address of the server')
parser.add_argument('-userid', type=str, default='user', help='user id')
parser.add_argument('-T', action='store_true', help='TEST mode')
parser.add_argument('-random', action='store_true', help='Random or not')

opt = parser.parse_args()

dataBuffer = Queue()
cmdBuffer = Queue()

TEST = opt.T
IP = opt.ip
PORT = 65432

th1 = EEGThread(0, opt.winsize, opt.stride)
th1.daemon = True
th2 = InferenceThread(1, IP, PORT)
th2.daemon = True
th3 = MitigationThread(2, 5)
th3.daemon = True

th1.start()
th2.start()
th3.start()

# TKinter stuff
mainwindow = tkinter.Tk()

mainwindow.title("N-back task Adaptive scheduling")
mainwindow.geometry("500x600")

title = tkinter.Label(mainwindow, text="Get ready", font=("Arial", 30))
title.pack()

feedback = tkinter.Label(mainwindow, text="", font=("Arial", 30))
feedback.pack()

# changeTitle(task_name_, tasks_)

number = tkinter.Label(mainwindow, text="-", font=("Arial", 300))
number.pack()

interval_ = 2.25
length_ = 60
tasks_ = [1,1,1,3,3,3]
random_ = opt.random

button = tkinter.Button(
    mainwindow, 
    text="Start",
    command= lambda: startTask(opt.userid, interval_, length_, tasks_, random_))
button.pack()

mainwindow.bind("<space>", check)

mainwindow.mainloop()