import tkinter
import time
import threading
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import os
import random
import argparse
import logging

# global variables
btnState = False # False when not clicked, True when clicked and in process
next_task = 0
curr_letter = ''
task_window = []
window_full = False

WAIT_FOR_KEY = False
BEST_REACT_TIME = None

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

class channelThread(threading.Thread):
    def __init__(self, tID, type_name, task_name, duration, starttime, path):
        threading.Thread.__init__(self)
        self.threadID = tID
        self.type_name = type_name
        self.task_name = task_name
        self.duration = duration
        self.starttime = starttime
        self.path = path
    def run(self):
        print("Starting {} receiver for {}, starting {} and last {}s"\
            .format(self.type_name, self.task_name, self.starttime, self.duration))
        try:
            streams = resolve_stream('type', self.type_name)
            inlet = StreamInlet(streams[0])
            filename = '{}_{}_{}_{}.txt'.format(self.task_name, self.type_name, self.starttime, self.duration)
            # logger = setup_logger('data', os.path.join(self.path, filename))
            # while time.time() - self.starttime <= self.duration:
            #     sample, timestamp = inlet.pull_sample()
            #     # list, float
            #     logger.info(str(timestamp) + ',' + ','.join([str(s) for s in sample]))
            with open(os.path.join(self.path, filename), 'w') as outfile:
                while time.time() - self.starttime <= self.duration:
                    sample, timestamp = inlet.pull_sample()
                    # list, float
                    outfile.write(str(timestamp) + ',' + ','.join([str(s) for s in sample]) + '\n')
        except KeyboardInterrupt as e:
            print("Ending program: EEG")
            raise e
        except Exception as e:
            print(e)
        finally:
            print("Exiting {} receiver for {}, starting {} and last {}s"\
                .format(self.type_name, self.task_name, self.starttime, self.duration))

def N_back_dynamic(file_path, interval, length, tasks):
    global WAIT_FOR_KEY, BEST_REACT_TIME, next_task, btnState
    logger = setup_logger('task', os.path.join(file_path, 'task_{}.log'.format(time.time())))
    # logging.basicConfig(filename=os.path.join(file_path, 'task_{}.log'.format(time.time())), 
    # level=logging.INFO)

    dash_int = 0.5

    logger.info('Task length={}, start!'.format(length))
    react_time = {}
    correct_react = {}

    react_time[tasks[next_task]] = []
    correct_react[tasks[next_task]] = 0

    curr_task = tasks[next_task]
    seq, res = randomTaskGenerator(curr_task, length)
    title.config(text='{}-back'.format(curr_task))
    logger.info('Task {}-back start!'.format(curr_task))
    for i in range(length):
        WAIT_FOR_KEY = True
        start_timer = time.time()
        number.config(text=seq[i])
        time.sleep(interval - dash_int)
        WAIT_FOR_KEY = False
        if res[i] == 1:
            if BEST_REACT_TIME == None:
                # feedback.config(text='False', fg='red')
                react_time[curr_task].append(interval - dash_int)
                logger.info('{},{},{},{}'.format(seq[i], res[i], False, interval - dash_int))
            else:
                # feedback.config(text='True', fg='light green')
                react_time[curr_task].append(BEST_REACT_TIME - start_timer)
                logger.info('{},{},{},{}'.format(seq[i], res[i], True, BEST_REACT_TIME - start_timer))
                correct_react[curr_task] += 1
        else:
            if BEST_REACT_TIME == None:
                # feedback.config(text='True', fg='light green')
                logger.info('{},{},{},{}'.format(seq[i], res[i], True, 'N/A'))
                correct_react[curr_task] += 1
            else:
                # feedback.config(text='False', fg='red')
                logger.info('{},{},{},{}'.format(seq[i], res[i], False, 'N/A'))
        BEST_REACT_TIME = None
        feedback.config(text='response recorded', fg='black')
        number.config(text='-')
        time.sleep(dash_int)
        feedback.config(text='', fg='black')
    results = loggingResult(correct_react, react_time)
    for r in results:
        logger.info(r)
    next_task += 1
    if (next_task >= len(tasks)):
        title.config(text='Finish!')
    else:
        title.config(text='Next task: {}-back...'.format(tasks[next_task]))
        btnState = False


def loggingResult(corrects, times):
    results = []
    total_times = []
    total_corrects = 0
    for key in corrects:
        curr_times = times[key]
        curr_corrects = corrects[key]
        total_times += curr_times
        total_corrects += curr_corrects
        avg_react_time = None if len(curr_times) == 0 else sum(curr_times)/len(curr_times)
        s = str(key) + '-back task: average react time: {}, correct react: {}'.format(avg_react_time, curr_corrects)
        results.append(s)
    avg_react_time = None if len(total_times) == 0 else sum(total_times)/len(total_times)
    s = 'Total average react time: {}, correct react: {}'.format(avg_react_time, total_corrects)

    results.append(s)
    return results

def randomTaskGenerator(N, length):
    '''
    make sure at least 10 matches in the sequence
    '''
    print('Generating sequence...')
    seq = [''] * length
    res = [0] * length
    letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # letters = ['B', 'F', 'H', 'J', 'L', 'M', 'Q', 'R', 'X']

    if length / N > 10:
        for i in range(10):
            pairhead = random.randint(0, length-1-N)
            while seq[pairhead] != '' or seq[pairhead + N] != '':
                pairhead = random.randint(0, length-1-N)
            num = random.randint(0, len(letters)-1)
            seq[pairhead] = letters[num]
            seq[pairhead + N] = letters[num]
    for i in range(length):
        if seq[i] == '':
            num = random.randint(0, len(letters)-1)
            seq[i] = letters[num]
    for i in range(N, length):
        if seq[i-N] == seq[i]:
            res[i] = 1
    
    print('Sequence generated!')
    return seq, res

def check(event=None):
    global WAIT_FOR_KEY, BEST_REACT_TIME
    if WAIT_FOR_KEY and BEST_REACT_TIME == None:
        BEST_REACT_TIME = time.time()

def startTask(user_id, task_name, interval, length, tasks):
    '''
    user_id::str        user id
    task_name::str      task name
    interval::float     between each number (2.25s by default)
    length::int         the length of the sequence
    tasks::[int]        list of tasks
    '''
    global btnState, next_task
    if btnState:
        print('already started a process, quit!')
        return
    btnState = True

    file_path = os.path.join('local', user_id)

    if not os.path.exists(file_path):
        os.mkdir(file_path)
    
    sub_task_name =  user_id + '_' + task_name + str(tasks[next_task])

    duration = interval * length

    # start recording
    th1 = channelThread(1, 'EEG', sub_task_name + '_' + str(length), duration, time.time(), file_path)
    th1.daemon = True
    th2 = channelThread(2, 'PPG', sub_task_name + '_' + str(length), duration, time.time(), file_path)
    th2.daemon = True
    th_task = threading.Thread(target=N_back_dynamic, args=(file_path, interval, length, tasks))
    th_task.daemon = True

    th1.start()
    th2.start()
    th_task.start()
    print('N-back thread is started!')


def changeTitle(task_name, tasks):
    global next_task
    sub_task_name = task_name + str(tasks[next_task])
    title.config(text=sub_task_name)

def simpleRecording(task_name, duration, user_id):
    if not os.path.exists(user_id):
        os.mkdir(user_id)
    duration *= 60
    th1 = channelThread(1, 'EEG', task_name, duration, time.time(), user_id)
    th2 = channelThread(2, 'PPG', task_name, duration, time.time(), user_id)
    th1.start()
    th2.start()
    print('recording started!')

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input for EEG data collector')
    parser.add_argument('-userid', type=str, default='xxx', help='user id')
    '''
    For simple recording 
    '''
    parser.add_argument('-simple', action='store_true', help='simply recording EEG signal')
    parser.add_argument('-taskname', type=str, default='', help='name of the task')
    parser.add_argument('-duration', type=int, default=5, help='duration(min) of the recording')
    '''
    For built-in n-back task
    '''
    parser.add_argument('-tasks', nargs='+', type=int, default=[1,2, 3], help='Tasks')
    parser.add_argument('-length', type=int, default=150, help='length of the sequence')
    parser.add_argument('-T', action='store_true', help='if the task is for training users')

    opt = parser.parse_args()

    if opt.simple:
        simpleRecording(opt.taskname, opt.duration, opt.userid)
    else:
        # N-back configs
        tasks_ = opt.tasks 
        random.shuffle(tasks_)
        task_name_ = 'T' if opt.T else 'R' # T for training, R for recorded
        user_id_ = opt.userid
        interval_ = 2.25
        length_ = opt.length

        # TKinter stuff
        mainwindow = tkinter.Tk()

        mainwindow.title("N-back task")
        mainwindow.geometry("500x600")

        title = tkinter.Label(mainwindow, text="test session", font=("Arial", 30))
        title.pack()

        feedback = tkinter.Label(mainwindow, text="", font=("Arial", 30))
        feedback.pack()
        
        title.config(text='First task: {}-back...'.format(tasks_[next_task]))

        number = tkinter.Label(mainwindow, text="-", font=("Arial", 300))
        number.pack()

        button = tkinter.Button(
            mainwindow, 
            text="Start",
            command= lambda: startTask(user_id_, task_name_, interval_, length_, tasks_))
        button.pack()

        mainwindow.bind("<space>", check)

        mainwindow.mainloop()