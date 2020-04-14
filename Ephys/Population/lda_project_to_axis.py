# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

LDA score of the population response between the two blocks and the actual probability left

@author: guido
"""

from os import listdir
from os.path import join
from itertools import groupby
import alf.io as ioalf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import brainbox as bb
import seaborn as sns
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from functions_5HT import download_data, paths, sessions

DOWNLOAD = False
PRE_TIME = 0.5
POST_TIME = 0

# Get all sessions
frontal_sessions, control_sessions = sessions()
all_ses = pd.concat((frontal_sessions, control_sessions), axis=0, ignore_index=True)
all_ses['recording'] = (['frontal']*frontal_sessions.shape[0]
                        + ['control']*control_sessions.shape[0])

DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'Population', 'LDA')
for i in range(all_ses.shape[0]):
    print('Performing LDA analysis [%d of %d]' % (i+1, all_ses.shape[0]))

    # Download data if required
    if DOWNLOAD is True:
        download_data(all_ses.loc[i, 'subject'], all_ses.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(DATA_PATH, all_ses.loc[i, 'lab'], 'Subjects',
                          all_ses.loc[i, 'subject'], all_ses.loc[i, 'date']))[0]
    session_path = join(DATA_PATH, all_ses.loc[i, 'lab'], 'Subjects',
                        all_ses.loc[i, 'subject'], all_ses.loc[i, 'date'], ses_nr)
    alf_path = join(DATA_PATH, all_ses.loc[i, 'lab'], 'Subjects', all_ses.loc[i, 'subject'],
                    all_ses.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(DATA_PATH, all_ses.loc[i, 'lab'], 'Subjects', all_ses.loc[i, 'subject'],
                      all_ses.loc[i, 'date'], ses_nr, 'alf', 'probe%s' % all_ses.loc[i, 'probe'])

    # Load in data
    spikes = ioalf.load_object(probe_path, 'spikes')
    clusters = ioalf.load_object(probe_path, 'clusters')
    trials = ioalf.load_object(alf_path, '_ibl_trials')

    # Only use single units
    spikes.times = spikes.times[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]
    spikes.clusters = spikes.clusters[np.isin(
            spikes.clusters, clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good'])]

    # Get spike counts for all trials from biased blocks
    trial_incl = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
    trial_times = trials.stimOn_times[trial_incl]
    times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
    spike_counts, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times,
                                                                  spikes.clusters, times)
    pop_vector = np.rot90(spike_counts)
    prob_left = trials.probabilityLeft[trial_incl]
    trial_blocks = (prob_left > 0.55).astype(int)

    # Initialize cross-validation
    block_lengths = [sum(1 for i in g) for k, g in groupby(prob_left)]
    blocks = np.repeat(np.arange(len(block_lengths)), block_lengths)

    # Perform LDA analysis
    lda_transform = np.zeros(pop_vector.shape[0])
    lda_class = np.zeros(pop_vector.shape[0])

    lda = LDA(n_components=1)
    cv = LeaveOneGroupOut().split(pop_vector, groups=blocks)
    # cv = KFold().split(pop_vector)
    # cv = LeaveOneOut().split(pop_vector)
    for train_index, test_index in cv:
        lda.fit(pop_vector[train_index], trial_blocks[train_index])
        lda_transform[test_index] = np.rot90(lda.transform(pop_vector[test_index]))[0]
        lda_class[test_index] = lda.predict(pop_vector[test_index])
    print('f1 score: %f' % np.round(f1_score(trial_blocks, lda_class), 2))

    # Convolve LDA projection with a 10 trial window
    lda_convolve = np.convolve(lda_transform, np.ones((10,))/10, mode='same')

    # Plot
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    sns.set(style="ticks", context="paper", font_scale=2)
    ax1.plot(np.arange(1, trial_times.shape[0]+1), lda_transform, color=[0.6, 0.6, 0.6])
    ax1.plot(np.arange(1, trial_times.shape[0]+1), lda_convolve, 'k', lw=3)
    ax1.set_ylabel('Position along LDA axis')
    ax1.set(ylim=[-6, 6], xlabel='Trials')
    ax12 = ax1.twinx()
    ax12.plot(np.arange(1, trial_times.shape[0]+1),
              trials.probabilityLeft[(trials.probabilityLeft > 0.55)
                                     | (trials.probabilityLeft < 0.45)], color='red', lw=3)
    ax12.set_ylabel('Probability of left trial', color='tab:red')
    ax12.tick_params(axis='y', colors='red')
    ax12.set(xlabel='Trials', ylim=[0, 1])

    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s' % (all_ses.loc[i, 'subject'],
                                          all_ses.loc[i, 'date'])))
    plt.close(fig)
