# muse

### resources:

- https://www.krigolsonlab.com/working-with-muse.html

- https://eegedu.com/

- https://www.hahack.com/codes/bci-and-muse-headband-development/

- https://sites.google.com/a/interaxon.ca/muse-developer-site/developer-getting-started-guide

### Data collection steps
1. **Preparation**: 
    1) Make sure you have Python3 installed.
    2) `pip install pylsl` and install other dependencies in `data/collector.py` (there shouldn't be any)
    3) On Windows: Install BlueMuse; On Mac: Connect to a BLED112 dongle, install uvicMuse

2. **First time user**: Connect Muse device to Muse app on your phone, and run a calibration(by simply starting a session, and quit after the calibration is done. If you want to do meditation, go for it). This is just to make sure you know **how to wear it properly**. If you've done calibration and used Muse, you can ignore this step.

3. **Teminate the Muse app** before connecting to BlueMuse (suppose you use windows) as this app might interfere with bluetooth conenction between Muse and BlueMuse.

4. Keep Muse device turned on, and open BlueMuse, you should be able to see your device on it. Click start streaming, you can see a little white window with data specs appears. Then you're ready to collect data. 

5. Arguments you need to provide for `data/collector.py`:
- `-userid`: typically includes your name, date, and session ID, e.g., bob1112_1
- `-T`: if you want to practice, you can set this to True by simply type this argument
- `-length`: the length of the sequence in each task. if you want to practice, you may want to specify this arg with a smaller number. Remember in data collection, the length would be 150 by default.
- `-tasks`: the tasks included in one data collection session. E.g., [1,2,3] means you have 1-back, 2-back, 3-back in this session. I don't recommend change this to anything other than default if possible. 

When you are ready to start, recommended example inputs are:

`cd` to `data/`

if you want to practice: `python collector.py -userid xxx1112_1 -T -length 30`

if you are ready to collect data: `python collector.py -userid xxx1112_2`


### Note:

1. In each data collection session, by default, you will have 3 tasks: 1-back, 2-back, 3-back, and each lasts about 5 minutes (150 numbers in each task). Between each tasks, you can take a rest and start the next one whenever you want, but don't rest too long.

2. Please **keep heaset on during one session** (from the first task to the last one until you see 'Finish' on UI).

3. Please **keep the little white window on the side** when you are doing the tasks and if you notice it disappears, you might need to recollect this particular task by specifying `-tasks` arg.

4. Please check the data collected in `data/local`. A valid EEG data file should contain around 86270 lines, if it contains far less than that, don't worry, just leave it there and recollect the data of that particular task again. 

5. After each task please rate the difficulty: 0 (very easy) -> 7 (very hard), and **write them down somewhere**.

### Model training
`main.py`
