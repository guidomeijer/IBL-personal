#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:48:48 2020

@author: guido
"""

from os import listdir
from os.path import join
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
import pandas as pd
import numpy as np
import seaborn as sns
from functions_5HT import download_data, paths, sessions, plot_settings

# Settings
DOWNLOAD = False
WIN_CENTERS = np.arange(-1, 2, 0.15)
WIN_SIZE = 0.2
DECODER = 'bayes'  # bayes, regression or forest
VALIDATION = 'kfold'

# Get all sessions
frontal_sessions, control_sessions = sessions()
all_ses = pd.concat((frontal_sessions, control_sessions), axis=0, ignore_index=True)
all_ses['recording'] = (['frontal']*frontal_sessions.shape[0]
                        + ['control']*control_sessions.shape[0])

DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'OverTime')
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

    # Get trial vectors
    incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
    trial_times = trials.goCue_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

    f1_scores = np.empty(WIN_CENTERS.shape)
    auroc = np.empty(WIN_CENTERS.shape)
    for j, win_center in enumerate(WIN_CENTERS):

        # Decode block identity for this time window
        decode_result = bb.population.decode(spikes.times, spikes.clusters,
                                             trial_times, trial_blocks,
                                             pre_time=-win_center+(WIN_SIZE/2),
                                             post_time=win_center+(WIN_SIZE/2),
                                             classifier=DECODER,
                                             cross_validation=VALIDATION,
                                             prob_left=probability_left)
        f1_scores[j] = decode_result['f1']
        auroc[j] = decode_result['auroc']

    # Add results to dataframe
    results = results.append(pd.DataFrame(data={
                'f1_scores': f1_scores, 'auroc': auroc, 'win_centers': WIN_CENTERS,
                'win_size': WIN_SIZE, 'subject': all_ses.loc[i, 'subject'],
                'date': all_ses.loc[i, 'date'], 'recording': all_ses.loc[i, 'recording']}))

# Save dataframe
results['session'] = results['subject'] + '_' + results['date']
results.to_csv(join(SAVE_PATH, 'Decoding', 'decoding_blocks_over_time.csv'))

# %% Plot

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
plot_settings()

sns.lineplot(x='win_centers', y='f1_scores', data=results[results['recording'] == 'frontal'],
             units='session', estimator=None, color=[0.7, 0.7, 0.7], ax=ax1)
sns.lineplot(x='win_centers', y='f1_scores', data=results[results['recording'] == 'frontal'],
             lw=2, ci=68, ax=ax1)
ax1.set(ylabel='Classification performance (F1 score)', xlabel='Time (s)',
        title='Decoding of block identity', ylim=[0, 1])

sns.lineplot(x='win_centers', y='f1_scores', data=results[results['recording'] == 'control'],
             units='session', estimator=None, color=[0.7, 0.7, 0.7], ax=ax2)
sns.lineplot(x='win_centers', y='f1_scores', data=results[results['recording'] == 'control'],
             lw=2, ci=68, ax=ax2)
ax2.set(ylabel='Classification performance (F1 score)', xlabel='Time (s)',
        title='Decoding of block identity', ylim=[0, 1])

sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'frontal'],
             units='session', estimator=None, color=[0.7, 0.7, 0.7], ax=ax3)
sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'frontal'],
             lw=2, ci=68, ax=ax3)
ax3.set(ylabel='Classification performance (AUROC)', xlabel='Time (s)',
        title='Frontal recordings', ylim=[0.2, 1])

sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'control'],
             units='session', estimator=None, color=[0.7, 0.7, 0.7], ax=ax4)
sns.lineplot(x='win_centers', y='auroc', data=results[results['recording'] == 'control'],
             lw=2, ci=68, ax=ax4)
ax4.set(ylabel='Classification performance (AUROC)', xlabel='Time (s)',
        title='Non-frontal recordings', ylim=[0.2, 1])

plt.tight_layout(pad=2)
sns.despine(trim=True)
plt.savefig(join(FIG_PATH, 'decoding_block_over_time'), dpi=300)
