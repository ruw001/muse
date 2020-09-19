import os
from sklearn import svm
import h5py
from scipy import signal
import numpy as np
from tqdm import tqdm
from scipy import signal as scisig
import matplotlib.pyplot as plt


def bandpower(data, sf, bands, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    bands : list of tuples
      Elements are lower and upper frequencies of the band of interest.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    from mne.time_frequency import psd_array_multitaper

    # Compute the modified periodogram (Welch)
    psd, freqs = psd_array_multitaper(
        data, sf, adaptive=True, normalization='full', verbose=0)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]
    bps = []
    for band in bands:
        low, high = band
        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            bp /= simps(psd, dx=freq_res)
        bps.append(bp)
    return bps


def makeNewInput(data, sf, bands):
    newdata = []
    for i in tqdm(range(data.shape[0])):
        newitem = []
        for c in range(data.shape[1]):
            res = bandpower(data[i, c, :], sf, bands)
            newitem += res
        newdata.append(newitem)
    newdata = np.array(newdata)
    assert newdata.shape == (data.shape[0], data.shape[1]*len(bands))
    return newdata

def svm_test():
  datapath = 'data/ru0718'
  type_ = 'EEG'
  winsize = 30
  stride = 0.1

  trainpath = os.path.join(datapath, 'train', 'train_{}_{}_{}.h5'.format(type_, winsize, stride))
  testpath = os.path.join(datapath, 'test', 'test_{}_{}_{}.h5'.format(type_, winsize, stride))

  trainfile = h5py.File(trainpath, 'r')
  testfile = h5py.File(testpath, 'r')

  traininput = trainfile['train']
  trainoutput = trainfile['train_labels']

  testinput = testfile['test']
  testoutput = testfile['test_labels']

  print(traininput.shape)
  print(testinput.shape)

  sf_ = 256
  win = 30*sf_
  bands_ = [(4, 8), (8, 12)] # theta, alpha

  print('extracting freq bands...')
  new_train = makeNewInput(traininput, sf_, bands_)
  new_test = makeNewInput(testinput, sf_, bands_)

  print('training...')
  clf = svm.SVC(decision_function_shape='ovo')
  clf.fit(new_train, trainoutput)
  print('done!')
  test_res = clf.predict(new_test)

  print('testing result is', (test_res == testoutput).sum()/len(test_res))

def z_score_test():
  from scipy import stats
  path = 'data/ru0802/1/ru0802_R1_30_EEG_1596419978.3037233_67.5.txt'
  signal = []
  with open(path, 'r') as infile:
    lines = infile.readlines()
    for l in lines:
        entries = l.split(',')
        # need to change if data format is different
        electrodes = [float(e) for e in entries[1:5]]
        signal.append(electrodes)
  signal = np.array(signal)
  signal = stats.zscore(signal, axis=0)
  ws = int(30 * 256)
  st = int(0.1 * 256)

  print(signal)
  print(signal.shape)

def stft_psd_extract(freq, data):
  '''
    data.shape: 30*256 x 4 
  '''
  psds = []
  for j in range(4):
    f, t, zxx = scisig.stft(data[:,j], fs=freq, nperseg=freq, nfft=freq, axis=0)
    # if j == 0:
    #   print(f, f.shape)
    #   print(t, t.shape)
    zxx = zxx[3:56, :]
    psds.append(np.abs(zxx))
  return np.array(psds)


def conv_2d_test():
  from scipy import stats
  path = 'data/ru0802/1/ru0802_R1_30_EEG_1596419978.3037233_67.5.txt'
  signal = []
  with open(path, 'r') as infile:
    lines = infile.readlines()
    for l in lines:
        entries = l.split(',')
        # need to change if data format is different
        electrodes = [float(e) for e in entries[1:5]]
        signal.append(electrodes)
  
  signal = np.array(signal)
  winsize = 30
  stride = 0.1
  freq = 256
  ws = int(winsize * freq)
  st = int(stride * freq)

  data = []

  for i in range(0, signal.shape[0], st):
    if i+ws > signal.shape[0]:
      data.append(stft_psd_extract(freq, signal[-ws:]))
      break
    else:
      data.append(stft_psd_extract(freq, signal[i:i+ws]))

  t = np.arange(0, 30.5, 0.5)
  f = np.arange(3, 56)
  print(np.array(data).shape)
  for psd in data:
    for i in range(4):
      plt.pcolormesh(t, f, psd[i,:], shading='gouraud')
      plt.title('STFT Magnitude')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      plt.show()

conv_2d_test()

