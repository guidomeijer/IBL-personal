# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

LDA score of the population response between the two blocks and the actual probability left

@author: guido
"""

from os import listdir
from os.path import join
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import LeaveOneOut
from functions_5HT import download_data, paths, sessions

DOWNLOAD = False
OVERWRITE = False
FRONTAL_CONTROL = 'Control'
PRE_TIME = 0.5
POST_TIME = 0

if FRONTAL_CONTROL == 'Frontal':
    sessions, _ = sessions()
elif FRONTAL_CONTROL == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'Population', 'QDA')
for i in range(sessions.shape[0]):
    # Download data if required
    if DOWNLOAD is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                          sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    session_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                        sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr)
    alf_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                    sessions.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                      sessions.loc[i, 'date'], ses_nr, 'alf', 'probe%s' % sessions.loc[i, 'probe'])

    # Load in data
    spikes = ioalf.load_object(probe_path, 'spikes')
    clusters = ioalf.load_object(probe_path, 'clusters')
    trials = ioalf.load_object(alf_path, '_ibl_trials')

    # Only use single units
    spikes.times = spikes.times[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]
    spikes.clusters = spikes.clusters[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]

    # Get spike counts for all trials
    trial_times = trials.stimOn_times[(trials.probabilityLeft > 0.55)
                                      | (trials.probabilityLeft < 0.45)]
    times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
    spike_counts, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times,
                                                                  spikes.clusters, times)
    trial_blocks = (trials.probabilityLeft[
                            (((trials.probabilityLeft > 0.55)
                              | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)

    # Transform to LDA with leave-one-out cross validation
    print('Projecting to LDA axis with leave-one-out cross-validation [%d of %d]' % (
                                                                        i+1, sessions.shape[0]))
    resp = np.rot90(spike_counts)
    loo = LeaveOneOut()
    qda_transform = np.zeros(resp.shape[0])
    for train_index, test_index in loo.split(resp):
        qda = QDA()
        qda.fit(resp[train_index], trial_blocks[train_index])
        qda_transform[test_index] = np.rot90(qda.transform(resp[test_index]))[0]
    qda_convolve = np.convolve(qda_transform, np.ones((10,))/10, mode='same')

    # Plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    sns.set(style="ticks", context="paper", font_scale=2)
    ax1.plot(np.arange(1, trial_times.shape[0]+1), qda_transform, color=[0.6, 0.6, 0.6])
    ax1.plot(np.arange(1, trial_times.shape[0]+1), qda_convolve, 'k', lw=3)
    ax1.set_ylabel('Position along LDA axis')
    ax1.set(ylim=[-6, 6])
    ax2 = ax1.twinx()
    ax2.plot(np.arange(1, trial_times.shape[0]+1),
             trials.probabilityLeft[(trials.probabilityLeft > 0.55)
                                    | (trials.probabilityLeft < 0.45)], color='red', lw=3)
    ax2.set_ylabel('Probability of left trial', color='tab:red')
    ax2.tick_params(axis='y', colors='red')
    ax2.set(xlabel='Trials', ylim=[0, 1])
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s_%s' % (FRONTAL_CONTROL, sessions.loc[i, 'subject'],
                                             sessions.loc[i, 'date'])))
    plt.close(fig)
