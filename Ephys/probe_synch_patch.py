# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:48:12 2019

@author: guido
"""

import numpy as np
from scipy import signal
from os.path import join
import matplotlib.pyplot as plt

# Settings
cam_channels = [2, 3, 4]  # Channel mapping
time_select = [500, 520]  # Recording time to process in seconds
sampling_freq = 30000  # Sampling frequency

# Define path to data
# path_probe00 = 'C:\\Users\\guido\\Google Drive\\TempData\\probe00'
# path_probe01 = 'C:\\Users\\guido\\Google Drive\\TempData\\probe01'
# path_probe00 = '/home/guido/IBLserver/Subjects/ZM_2407/2019-11-06/001/raw_ephys_data/probe_00'
# path_probe01 = '/home/guido/IBLserver/Subjects/ZM_2407/2019-11-06/001/raw_ephys_data/probe_01'
# path_probe00 = '/home/guido/IBLserver/Subjects/ZM_2407/2019-11-05/002/raw_ephys_data/probe_00'
# path_probe01 = '/home/guido/IBLserver/Subjects/ZM_2407/2019-11-05/002/raw_ephys_data/probe_01'
path_probe00 = '/home/guido/IBLserver/Subjects/ZM_2406/2019-11-12/001/raw_ephys_data/probe00'
path_probe01 = '/home/guido/IBLserver/Subjects/ZM_2406/2019-11-12/001/raw_ephys_data/probe01'

# PROBE 00
channels_00 = np.load(join(path_probe00, '_spikeglx_sync.channels.probe00.npy'))
polarities_00 = np.load(join(path_probe00, '_spikeglx_sync.polarities.probe00.npy'))
timestamps_00 = np.load(join(path_probe00, '_spikeglx_sync.times.probe00.npy'))

# Select only part of the recording
channels_00 = channels_00[(timestamps_00 > time_select[0]) & (timestamps_00 < time_select[1])]
polarities_00 = polarities_00[(timestamps_00 > time_select[0]) & (timestamps_00 < time_select[1])]
timestamps_00 = timestamps_00[(timestamps_00 > time_select[0]) & (timestamps_00 < time_select[1])]

# Reconstruct trace
time_00 = np.arange(time_select[0], time_select[1], 1/sampling_freq)
sync_00 = np.zeros(np.size(time_00))
for i, cam in enumerate(cam_channels):
    for j, ts in enumerate(timestamps_00[channels_00 == cam]):
        sync_00[np.argmin(np.abs(time_00 - ts))] = polarities_00[timestamps_00 == ts][0] * cam

# PROBE 01
channels_01 = np.load(join(path_probe01, '_spikeglx_sync.channels.probe01.npy'))
polarities_01 = np.load(join(path_probe01, '_spikeglx_sync.polarities.probe01.npy'))
timestamps_01 = np.load(join(path_probe01, '_spikeglx_sync.times.probe01.npy'))

# Select only part of the recording
channels_01 = channels_01[(timestamps_01 > time_select[0]) & (timestamps_01 < time_select[1])]
polarities_01 = polarities_01[(timestamps_01 > time_select[0]) & (timestamps_01 < time_select[1])]
timestamps_01 = timestamps_01[(timestamps_01 > time_select[0]) & (timestamps_01 < time_select[1])]

# Reconstruct trace
time_01 = np.arange(time_select[0], time_select[1], 1/sampling_freq)
sync_01 = np.zeros(np.size(time_01))
for i, cam in enumerate(cam_channels):
    for j, ts in enumerate(timestamps_01[channels_01 == cam]):
        sync_01[np.argmin(np.abs(time_01 - ts))] = polarities_01[timestamps_01 == ts][0] * cam

# Get lag with cross-correlation
corr = np.correlate(sync_00, sync_01, 'full')
lag = int(np.ceil(np.argmax(corr) - np.size(corr)/2))

# Get first pulse
if lag < 0:
    print('Probe 00 started %.2f seconds after probe 01' % (time_00[np.abs(lag)]-time_select[0]))
    first_pulse = np.argmin(np.abs(timestamps_00 - time_00[np.abs(lag)]))
    probe_started_first = '01'

    plt.plot(time_01, sync_01)
    plt.plot(time_00 + (time_00[np.abs(lag)]-time_select[0]), sync_00)

elif lag > 0:
    print('Probe 00 started %.2f seconds after probe 01' % (time_00[np.abs(lag)]-time_select[0]))
    probe_started_first = '00'

    plt.plot(time_00, sync_00)
    plt.plot(time_01 + (time_01[lag]-time_select[0]), sync_01)

