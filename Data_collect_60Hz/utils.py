import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error
import math
from scipy.signal import savgol_filter, medfilt
# import jenkspy

from scipy import stats

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def remove_zero(data):
    seq_out = np.zeros((data.shape[0], 2))
    nozero = np.squeeze(np.array(np.nonzero(data[:, 0])))

    print('nozero', nozero.shape)
    if not nozero.shape:
        return data
    if nozero.shape[0]==0:
        return data
    print('remove_zero', data.shape)
    for i in range(0, data.shape[0]):
            if i not in np.array(nozero).tolist():
                ner = find_nearest(nozero, i)
                seq_out[i] = data[ner]
            else:
                seq_out[i] = data[i]
    return seq_out


def skeleton_filter(skeleton_seq):
    data = np.squeeze(skeleton_seq[:, 0, :])
    data = remove_zero(data)

    data[:, 0] = medfilt(data[:, 0], 9)
    data[:, 1] = medfilt(data[:, 1], 9)
    seq_out = np.zeros((data.shape[0], 2))
    seq_out[:, 0] = savgol_filter(data[:, 0], 15, 3, mode='interp')
    seq_out[:, 1] = savgol_filter(data[:, 1], 15, 3, mode='interp')

    # draw_t_save_hand(seq_out)


    return seq_out


def draw_t_save_hand(data):
    tip = data[10:-10]
    plt.plot(tip.T[0], tip.T[1], 'o-')
    plt.grid(False)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.show()


