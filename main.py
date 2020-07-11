import tkinter
import time
import threading
from muselsl import stream, list_muses, view, record
from multiprocessing import Process
from mne import Epochs, find_events
from time import time, strftime, gmtime
import os
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

def eeg_retriever():
    muse = list_muses()[0]
    stream_process = Process(target=stream, args=(muse,))
    stream_process.start()
    view()

def test():
    time.sleep(4)
    print("hello")
    # label.config(text="changed")

# TKinter stuff
# mainwindow = tkinter.Tk()
#
# mainwindow.title("N-back task")
#
# label = tkinter.Label(mainwindow, text="test123123123", font=("Arial", 24))
# label.pack()
#
# button = tkinter.Button(mainwindow, text="test", command=threading.Thread(target=test).start())
# button.pack()
#
# mainwindow.mainloop()

eeg_retriever()