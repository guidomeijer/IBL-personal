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
from ephys_functions import download_data, paths, frontal_sessions

download = False
sessions = frontal_sessions()
T_BIN = 0.01

DATA_PATH, FIG_PATH = paths()
FIG_PATH = '/home/guido/Figures/Ephys/'
for i in range(sessions.shape[0]):
    # Download data if required
    if download is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(DATA_PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    session_path = join(DATA_PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr)
    alf_path = join(DATA_PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(DATA_PATH, sessions.loc[i, 'subject'], sessions.loc[i, 'date'],
                      ses_nr, 'alf', 'probe%s' % sessions.loc[i, 'probe'])

    # Load in data
    spikes = ioalf.load_object(probe_path, 'spikes')
    clusters = ioalf.load_object(probe_path, 'clusters')
    trials = ioalf.load_object(alf_path, '_ibl_trials')

    # Only use single units
    spikes.times = spikes.times[np.isin(
        spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]
    spikes.clusters = spikes.clusters[np.isin(
        spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]

    for n, cluster in enumerate(spikes.clusters):
        fig = plt.figure()
        bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                          trials.stimOn_times[trials.contrastLeft == 0],
                                          cluster, t_before=1, t_after=2, error_bars='sem',
                                          include_raster=True)
        plt.title('Stimulus onset')
        plt.savefig(join(FIG_PATH, 'PSTH', '%s' % sessions.loc[i, 'subject'],
                         '%s' % sessions.loc[i, 'date'],
                         'p%s_n%s' % (sessions.loc[i, 'probe'], cluster)))
        plt.close(fig)


