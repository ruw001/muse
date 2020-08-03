import tkinter
import time
import threading
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
import os
import random

# global variables
btnState = False # False when not clicked, True when clicked and in process
curr_task = 0

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
            print("Ending program")
            raise e
        print("Exiting {} receiver for {}, starting {} and last {}s"\
            .format(self.type_name, self.task_name, self.starttime, self.duration))
        
def N_back(interval, length, tasks, task_name):
    global btnState, curr_task
    dash_int = 0.5
    for i in range(length):
        num = random.randint(0,9)
        number.config(text=str(num))
        time.sleep(interval-dash_int)
        number.config(text='-')
        time.sleep(dash_int)
    curr_task += 1
    if curr_task >= len(tasks):
        title.config(text='Finish!')
    else:
        changeTitle(task_name, tasks)
        btnState = False
    number.config(text='-')
    print("N-back thread is finished!")

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


if  __name__ == "__main__":
    # N-back configs
    tasks_ = [1,2,3,4,5]
    random.shuffle(tasks_)
    task_name_ = 'R' # T for training, R for recorded
    user_id_ = 'ru0802r5'#'x'
    interval_ = 2.25
    length_ = 30

    lock = threading.Lock()

    # TKinter stuff
    mainwindow = tkinter.Tk()

    mainwindow.title("N-back task")
    mainwindow.geometry("500x600")

    title = tkinter.Label(mainwindow, text="test session", font=("Arial", 30))
    title.pack()
    
    changeTitle(task_name_, tasks_)

    number = tkinter.Label(mainwindow, text="-", font=("Arial", 300))
    number.pack()

    button = tkinter.Button(
        mainwindow, 
        text="Start",
        command= lambda: startTask(user_id_, task_name_, interval_, length_, tasks_))
    button.pack()

    mainwindow.mainloop()