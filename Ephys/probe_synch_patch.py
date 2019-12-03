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
time_select = [0, 15]  # Recording time to process in seconds
sampling_freq = 30000  # Sampling frequency

# Define path to data
path_probe00 = 'C:\\Users\\guido\\Google Drive\\TempData\\probe00'
path_probe01 = 'C:\\Users\\guido\\Google Drive\\TempData\\probe01'

# PROBE 00
channels = np.load(join(path_probe00, '_spikeglx_sync.channels.probe_00.npy'))
polarities = np.load(join(path_probe00, '_spikeglx_sync.polarities.probe_00.npy'))
times = np.load(join(path_probe00, '_spikeglx_sync.times.probe_00.npy'))

# Select only first part of recording
channels = channels[(times > time_select[0]) & (times < time_select[1])]
polarities = polarities[(times > time_select[0]) & (times < time_select[1])]
times = times[(times > time_select[0]) & (times < time_select[1])]

# Reconstruct trace
time_00 = np.arange(time_select[0], time_select[1], 1/sampling_freq)
sync_00 = np.zeros(np.size(time_00))
for i, cam in enumerate(cam_channels):
    for j, ts in enumerate(times[channels == cam]):
        sync_00[np.argmin(np.abs(time_00 - ts))] = polarities[times == ts][0] * cam

# PROBE 01
channels = np.load(join(path_probe01, '_spikeglx_sync.channels.probe_01.npy'))
polarities = np.load(join(path_probe01, '_spikeglx_sync.polarities.probe_01.npy'))
times = np.load(join(path_probe01, '_spikeglx_sync.times.probe_01.npy'))

# Select only first part of recording
channels = channels[(times > time_select[0]) & (times < time_select[1])]
polarities = polarities[(times > time_select[0]) & (times < time_select[1])]
times = times[(times > time_select[0]) & (times < time_select[1])]

# Reconstruct trace
time_01 = np.arange(time_select[0], time_select[1], 1/sampling_freq)
sync_01 = np.zeros(np.size(time_01))
for i, cam in enumerate(cam_channels):
    for j, ts in enumerate(times[channels == cam]):
        sync_01[np.argmin(np.abs(time_01 - ts))] = polarities[times == ts][0] * cam

# Get lag with cross-correlation
npcorr = np.correlate(sync_00, sync_01, 'full')
lag = np.argmax(npcorr)


