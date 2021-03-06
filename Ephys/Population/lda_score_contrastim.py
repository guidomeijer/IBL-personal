# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import listdir
from os.path import join, isfile
import alf.io as ioalf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from functions_5HT import (download_data, paths, sessions, decoding, plot_settings,
                           get_spike_counts_in_bins)

# Settings
DOWNLOAD = False
FRONTAL_CONTROL = 'Frontal'
DEPTH_BIN_CENTERS = np.arange(200, 4000, 200)
DEPTH_BIN_SIZE = 300
PRE_TIME = 0
POST_TIME = 0.5
MIN_CONTRAST = 0.1

if FRONTAL_CONTROL == 'Frontal':
    sessions, _ = sessions()
elif FRONTAL_CONTROL == 'Control':
    _, sessions = sessions()

DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Population', 'LDA_ContraStim')
for i in range(sessions.shape[0]):
    print('Starting subject %s, session %s' % (sessions.loc[i, 'subject'],
                                               sessions.loc[i, 'date']))

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
            spikes.clusters, clusters.metrics.cluster_id[
                                clusters.metrics.ks2_label == 'good'])]
    spikes.clusters = spikes.clusters[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[
                                clusters.metrics.ks2_label == 'good'])]
    clusters.channels = clusters.channels[clusters.metrics.ks2_label == 'good']
    clusters.depths = clusters.depths[clusters.metrics.ks2_label == 'good']
    cluster_ids = clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good']

    # Get trial indices
    inconsistent = (((trials.probabilityLeft > 0.55)
                     & (trials.contrastRight > MIN_CONTRAST))
                    | ((trials.probabilityLeft < 0.45)
                       & (trials.contrastLeft > MIN_CONTRAST)))
    consistent = (((trials.probabilityLeft > 0.55)
                   & (trials.contrastLeft > MIN_CONTRAST))
                  | ((trials.probabilityLeft < 0.45)
                     & (trials.contrastRight > MIN_CONTRAST)))
    trial_times = trials.stimOn_times[(consistent == 1) | (inconsistent == 1)]
    trial_consistent = np.zeros(consistent.shape[0])
    trial_consistent[consistent == 1] = 1
    trial_consistent[inconsistent == 1] = 2
    trial_consistent = trial_consistent[(consistent == 1) | (inconsistent == 1)]

    # Get matrix of all neuronal responses
    times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
    resp, cluster_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, times)
    resp = np.rot90(resp)

    # Decode block identity
    lda_score = np.empty(len(DEPTH_BIN_CENTERS))
    n_clusters = np.empty(len(DEPTH_BIN_CENTERS))
    for j, depth in enumerate(DEPTH_BIN_CENTERS):
        print('Decoding block identity from depth %d..' % depth)
        depth_clusters = cluster_ids[((clusters.depths > depth-(DEPTH_BIN_SIZE/2))
                                      & (clusters.depths < depth+(DEPTH_BIN_SIZE/2)))]
        if len(depth_clusters) <= 2:
            n_clusters[j] = len(depth_clusters)
            lda_score[j] = np.nan
            continue
        lda = LDA(n_components=1)
        lda_transform = lda.fit_transform(resp[:, np.isin(cluster_ids, depth_clusters)],
                                          trial_consistent)
        lda_score[j] = np.mean(np.abs(lda_transform))
        n_clusters[j] = len(depth_clusters)

    # Plot decoding versus depth
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
    ax1.plot(lda_score, DEPTH_BIN_CENTERS, lw=2)
    ax1.set(ylabel='Depth (um)', xlabel='LDA score',
            title='LDA')
    # for j, (x, y) in enumerate(zip(f1_over_shuffled[significant_depth],
    #                                DEPTH_BIN_CENTERS[significant_depth])):
    #   ax1.text(x+0.02, y+30, '*', va='center')
    ax2.plot(n_clusters, DEPTH_BIN_CENTERS, lw=2)
    ax2.set(xlabel='Number of neurons')
    ax2.invert_yaxis()
    plot_settings()
    plt.savefig(join(FIG_PATH, '%s_%s' % (sessions.loc[i, 'subject'],
                                          sessions.loc[i, 'date'])))
    plt.close(f)
