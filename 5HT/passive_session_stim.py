#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:28:56 2021

@author: guido
"""

import numpy as np
from os.path import join
from brainbox.plot import peri_event_time_histogram
import matplotlib.pyplot as plt
from ibllib.ephys.sync_probes import apply_sync

DATA_DIR = '/home/guido/FlatIron/ZFM-01507_2020-11-06/raw_ephys_data'
PLOT_DIR = '/home/guido/Figures/5HT/passive_laser_neurons'
T_BEFORE = 1
T_AFTER = 5
BIN_SIZE = 0.05

# Load in pulses
pulse_channels = np.load(join(DATA_DIR, '_spikeglx_sync.channels.npy'))
pulse_polarities = np.load(join(DATA_DIR, '_spikeglx_sync.polarities.npy'))
pulse_times = np.load(join(DATA_DIR, '_spikeglx_sync.times.npy'))

# Get laser times
laser_times = pulse_times[(pulse_channels == 17) & (pulse_polarities == 1)]
train_times = laser_times[np.append(5, np.diff(laser_times)) > 1]

# Load in spikes
spike_times = np.load(join(DATA_DIR, 'probe00', 'spike_times.npy')) / 10000
spike_clusters = np.load(join(DATA_DIR, 'probe00', 'spike_clusters.npy'))

for i, cluster in enumerate(np.unique(spike_clusters)):
    try:
        p, ax = plt.subplots()
        peri_event_time_histogram(spike_times, spike_clusters, train_times, cluster,
                                  t_before=T_BEFORE, t_after=T_AFTER, bin_size=BIN_SIZE,
                                  include_raster=True, error_bars='sem', ax=ax)
        plt.savefig(join(PLOT_DIR, f'{i}'))
        plt.close(p)
    except:
        print('error')


