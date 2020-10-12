import tkinter
import time
import threading
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import os
import random
import argparse

# global variables
btnState = False # False when not clicked, True when clicked and in process
curr_task = 0
curr_letter = ''
task_window = []
window_full = False

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
            with open(os.path.join(self.path, filename), 'w') as outfile:
                while time.time() - self.starttime <= self.duration:
                    sample, timestamp = inlet.pull_sample()
                    # list, float
                    outfile.write(str(timestamp) + ',' + ','.join([str(s) for s in sample]) + '\n')
        except KeyboardInterrupt as e:
            print("Ending program: EEG")
            raise e
        finally:
            print("Exiting {} receiver for {}, starting {} and last {}s"\
                .format(self.type_name, self.task_name, self.starttime, self.duration))
        
def N_back(interval, length, tasks, task_name):
    global btnState, curr_task, curr_letter, task_window, window_full
    try:
        dash_int = 0.5
        letters = ['B', 'F', 'H', 'J', 'L', 'M', 'Q', 'R', 'X']
        for i in range(length):
            num = random.randint(0,8)
            letter = letters[num]
            task_window.append(letter)
            if len(task_window) == tasks[curr_task]+1:
                window_full = True
            elif len(task_window) > tasks[curr_task]+1:
                task_window.pop(0)
            number.config(text=str(letter))
            time.sleep(interval-dash_int)
            number.config(text='-')
            time.sleep(dash_int)
            feedback.config(text='', fg='black')
        curr_task += 1
        if curr_task >= len(tasks):
            title.config(text='Finish!')
        else:
            changeTitle(task_name, tasks)
            btnState = False
        number.config(text='-')
        feedback.config(text='', fg='black')
        task_window = []
        window_full = False
        print("N-back thread is finished!")
    except KeyboardInterrupt as e:
        print("Ending program: N-back")
        raise e
    finally:
        print('Finish!')

def check(event=None):
    global task_window, window_full
    if len(task_window) == 0:
        return
    if task_window[0] == task_window[-1] and window_full:
        feedback.config(text='True', fg='light green')
    else:
        feedback.config(text='False', fg='red')

def startTask(user_id, task_name, interval, length, tasks):
    '''
    user_id::str        user id
    task_name::str      task name
    interval::float     between each number (2.25s by default)
    length::int         the length of the sequence
    tasks::[int]        list of tasks
    '''
    global btnState, curr_task
    if btnState:
        print('already started a process, quit!')
        return
    btnState = True

    if not os.path.exists(user_id):
        os.mkdir(user_id)
    
    sub_task_name =  user_id + '_' + task_name + str(tasks[curr_task])

    duration = interval * length

    # start recording
    th1 = channelThread(1, 'EEG', sub_task_name + '_' + str(length), duration, time.time(), user_id)
    th2 = channelThread(2, 'PPG', sub_task_name + '_' + str(length), duration, time.time(), user_id)
    th_task = threading.Thread(target=N_back, args=(interval, length, tasks, task_name))
    
    th1.start()
    th2.start()
    th_task.start()
    print('N-back thread is started!')


def changeTitle(task_name, tasks):
    global curr_task
    sub_task_name = task_name + str(tasks[curr_task])
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
    parser.add_argument('-tasks', nargs='+', type=int, default=[1,2,3], help='Tasks')
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
        
        changeTitle(task_name_, tasks_)

        number = tkinter.Label(mainwindow, text="-", font=("Arial", 300))
        number.pack()

        button = tkinter.Button(
            mainwindow, 
            text="Start",
            command= lambda: startTask(user_id_, task_name_, interval_, length_, tasks_))
        button.pack()

        mainwindow.bind("<space>", check)

        mainwindow.mainloop()