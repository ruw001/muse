import os
from sklearn import svm
import h5py
from scipy import signal
import numpy as np
from tqdm import tqdm


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







