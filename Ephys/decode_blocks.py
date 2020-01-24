# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir
from os.path import join
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
import ibllib.plots as iblplt

from brainbox.processing import bincount2D
import numpy as np
from ephys_functions import download_data, data_path, frontal_sessions

download = True
sessions = frontal_sessions()
T_BIN = 0.01

PATH = data_path()
for i in range(sessions.shape[0]):
    # Download data if required
    if download is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    session_path = join(PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr)
    alf_path = join(PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'],
                      ses_nr, 'alf', 'probe%s' % sessions.loc[i, 'probe'])

    # Load in data
    spikes = ioalf.load_object(probe_path, 'spikes')
    trials = ioalf.load_object(alf_path, '_ibl_trials')

    peth, bs = bb.plot.peths(spikes.times, spikes.clusters, 5, trials.stimOn_times)



    """
    R, times, clusters = bincount2D(spikes['times'], spikes['clusters'], T_BIN)
    plt.imshow(R, aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
               extent=np.r_[times[[0, -1]], clusters[[0, -1]]], origin='lower')

    # plot trial start and reward time
    reward = trials['feedback_times'][trials['feedbackType'] == 1]
    iblplt.vertical_lines(trials['intervals'][:, 0], ymin=0, ymax=clusters[-1],
                          color='k', linewidth=0.5, label='trial starts')
    iblplt.vertical_lines(reward, ymin=0, ymax=clusters[-1], color='m', linewidth=0.5,
                          label='valve openings')
    plt.xlim([0, 200])
    plt.xlabel('Time (s)')
    plt.ylabel('Cluster #')
    plt.legend()

    bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters, trials.goCue_times, 20,
                                      t_before=0.25, t_after=0.25, include_raster=True)
    """



