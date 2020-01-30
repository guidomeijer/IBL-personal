# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir, mkdir
from os.path import join, isdir
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
import numpy as np
from ephys_functions import download_data, paths, frontal_sessions

download = False
sessions = frontal_sessions()

DATA_PATH, FIG_PATH = paths()
FIG_PATH = join(FIG_PATH, 'PSTH')
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

    # Only use responsive neurons
    resp_neurons = bb.task.responsive_units(spikes.times, spikes.clusters, trials.goCue_times)[0]
    spikes.times = spikes.times[np.isin(spikes.clusters, resp_neurons)]
    spikes.clusters = spikes.clusters[np.isin(spikes.clusters, resp_neurons)]

    # Calculate whether neuron discriminates
    trial_times = trials.goCue_times[((trials.probabilityLeft > 0.55)
                                      | (trials.probabilityLeft < 0.55))]
    trial_blocks = (trials.probabilityLeft[((trials.probabilityLeft > 0.55)
                                            | (trials.probabilityLeft < 0.55))] > 0.55).astype(int)

    auc_roc, cluster_ids, auc_sig = bb.task.calculate_roc(spikes.times, spikes.clusters, trial_times,
                                                          trial_blocks, pre_time=0.5, post_time=0,
                                                          bootstrap=True, n_bootstrap=100)

    auc_sig = cluster_ids[(auc_roc > 0.65) | (auc_roc < 0.35)]

    # Make directories
    if not isdir(join(FIG_PATH, '%s' % sessions.loc[i, 'subject'])):
        mkdir(join(FIG_PATH, '%s' % sessions.loc[i, 'subject']))
    if not isdir(join(FIG_PATH, '%s' % sessions.loc[i, 'subject'],
                      '%s' % sessions.loc[i, 'date'])):
        mkdir(join(FIG_PATH, '%s' % sessions.loc[i, 'subject'], '%s' % sessions.loc[i, 'date']))
    if not isdir(join(FIG_PATH, '%s' % sessions.loc[i, 'subject'],
                      '%s' % sessions.loc[i, 'date'], 'blocks')):
        mkdir(join(FIG_PATH, '%s' % sessions.loc[i, 'subject'], '%s' % sessions.loc[i, 'date'],
                   'blocks'))

    for n, cluster in enumerate(auc_sig):
        fig, ax = plt.subplots(1, 1)
        bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                          trials.goCue_times[((trials.probabilityLeft > 0.5)
                                                              & (trials.choice == -1))],
                                          cluster, t_before=1, t_after=2, error_bars='sem', ax=ax)
        bb.plot.peri_event_time_histogram(spikes.times, spikes.clusters,
                                          trials.goCue_times[((trials.probabilityLeft < 0.5)
                                                              & (trials.choice == -1))],
                                          cluster, t_before=1, t_after=2, error_bars='sem',
                                          pethline_kwargs={'color': 'red', 'lw': 2},
                                          errbar_kwargs={'color': 'red', 'alpha': 0.5}, ax=ax)
        plt.legend(['Left block', 'Right block'])
        plt.title('t=0: go cue')
        plt.savefig(join(FIG_PATH,
                         '%s' % sessions.loc[i, 'subject'],
                         '%s' % sessions.loc[i, 'date'],
                         'significant_roc',
                         'p%s_n%s' % (sessions.loc[i, 'probe'], cluster)))
        plt.close(fig)
