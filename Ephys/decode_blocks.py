# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir
from os.path import join
import numpy as np
from ephys_functions import download_data, data_path, frontal_sessions

download = False
sessions = frontal_sessions()

PATH = data_path()
for i in range(sessions.shape[0]):
    # Download data if required
    if download is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Load in data from local drive
    ses_nr = listdir(join(PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    alf_path = join(PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr, 'alf')
    spike_times = np.load(join(alf_path, 'probe%s' % sessions.loc[i, 'probe'],
                               'spikes.times.npy'))
    spike_clusters = np.load(join(alf_path, 'probe%s' % sessions.loc[i, 'probe'],
                                  'spikes.clusters.npy'))
    prob_left = np.load(join(alf_path, '_ibl_trials.probabilityLeft.npy'))
    stim_times = np.load(join(alf_path, '_ibl_trials.stimOn_times.npy'))
    contrast_l = np.load(join(alf_path, '_ibl_trials.contrastLeft.npy'))
    contrast_r = np.load(join(alf_path, '_ibl_trials.contrastRight.npy'))




