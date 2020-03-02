#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:48:48 2020

@author: guido
"""

from os import listdir
from os.path import join, isfile
import alf.io as ioalf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import (download_data, paths, sessions, decoding, plot_settings,
                           get_spike_counts_in_bins)

# Settings
DOWNLOAD = False
FRONTAL_CONTROL = 'Frontal'
WIN_CENTERS = np.arange(-1, 2, 0.08)
WIN_SIZE = 0.1
DECODER = 'forest'  # bayes, regression or forest
NUM_SPLITS = 1
MIN_CONTRAST = 0.1

# Get all sessions
frontal_sessions, control_sessions = sessions()
all_ses = pd.concat((frontal_sessions, control_sessions), axis=0, ignore_index=True)
all_ses['recording'] = (['frontal']*frontal_sessions.shape[0]
                        + ['control']*control_sessions.shape[0])

# Initialize decoder
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'BlocksOverTime')
results = pd.DataFrame()
for i in range(all_ses.shape[0]):
    print('Starting subject %s, session %s' % (all_ses.loc[i, 'subject'],
                                               all_ses.loc[i, 'date']))

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

    f1_scores = np.empty(WIN_CENTERS.shape)
    auroc = np.empty(WIN_CENTERS.shape)
    for j, win_center in enumerate(WIN_CENTERS):

        # Get matrix of neuronal responses
        times = np.column_stack(((trial_times + (win_center-(WIN_SIZE/2))),
                                 (trial_times + (win_center+(WIN_SIZE/2)))))
        resp, cluster_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, times)
        resp = np.rot90(resp)

        # Decode block identity for this time window
        f1_scores[j], auroc[j], _ = decoding(resp, trial_consistent, clf, NUM_SPLITS)

    # Add results to dataframe
    results = results.append(pd.DataFrame(data={
                'f1_scores': f1_scores, 'auroc': auroc, 'win_centers': WIN_CENTERS,
                'win_size': WIN_SIZE, 'subject': all_ses.loc[i, 'subject'],
                'date': all_ses.loc[i, 'date'], 'recording': all_ses.loc[i, 'recording']}))

results['session'] = results['subject'] + '_' + results['date']
results.to_csv(join(SAVE_PATH, 'Decoding', 'decoding_contrastim_over_time.csv'))

# %% Plot

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
plot_settings()

sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'frontal'],
             units='session', estimator=None, color=[0.7, 0.7, 0.7], ax=ax1)
sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'frontal'],
             lw=2, ci=68, ax=ax1)
ax1.plot([np.min(WIN_CENTERS), np.max(WIN_CENTERS)], [0.5, 0.5], linestyle='--', color='k')
ax1.set(ylabel='Classification performance (AUROC)', xlabel='Time (s)',
        title='Frontal recordings', ylim=[0.35, 0.65])

sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'control'],
             units='session', estimator=None, color=[0.7, 0.7, 0.7], ax=ax2)
sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'control'],
             lw=2, ci=68, ax=ax2)
ax2.plot([np.min(WIN_CENTERS), np.max(WIN_CENTERS)], [0.5, 0.5], linestyle='--', color='k')
ax2.set(ylabel='Classification performance (AUROC)', xlabel='Time (s)',
        title='Non-frontal recordings', ylim=[0.35, 0.65])

sns.lineplot(x='win_centers', y='auroc', data=results, hue='recording', ci=68, lw=2, ax=ax3)
ax3.plot([np.min(WIN_CENTERS), np.max(WIN_CENTERS)], [0.5, 0.5], linestyle='--', color='k')
ax3.set(ylabel='Classification performance (AUROC)', xlabel='Time (s)', ylim=[0.4, 0.605])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(FIG_PATH, 'decoding_contrastim_over_time'), dpi=300)
