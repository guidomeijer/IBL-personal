#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Decode left/right block identity from all superficial recordings

@author: guido
"""

from os import listdir
from os.path import join, isfile
import alf.io as ioalf
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import decode
import pandas as pd
import seaborn as sns
import alf.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import (download_data, paths, sessions, decoding, plot_settings,
                           get_spike_counts_in_bins, one_session_path)
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 0.6
POST_TIME = -0.2
DECODER = 'bayes'  # bayes, regression or forest
VALIDATION = 'kfold'
NUM_SPLITS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Blocks')

# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times',
                            task_protocol='_iblrig_tasks_ephysChoiceWorld', details=True)

decoding_result = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
    probes = one.load_object(eid, 'probes', download_only=False)
    if (not hasattr(trials, 'stimOn_times')
            or (trials.stimOn_times.shape[0] != trials.probabilityLeft.shape[0])
            or (not hasattr(probes, 'trajectory'))):
        continue
    for p in range(len(probes['trajectory'])):
        # Select shallow penetrations
        if probes['trajectory'][p]['phi'] == 180:
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                continue

            # Only use good single units
            clusters_to_use = clusters.metrics.ks2_label == 'good'
            if clusters_to_use.sum() < 4:
                continue
            spikes.times = spikes.times[
                    np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
            spikes.clusters = spikes.clusters[
                    np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
            clusters.depths = clusters.depths[clusters_to_use]
            cluster_ids = clusters.metrics.cluster_id[clusters_to_use]

            # Get trial vectors
            incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
            trial_times = trials.goCue_times[incl_trials]
            if trial_times.shape[0] < 400:
                continue
            probability_left = trials.probabilityLeft[incl_trials]
            trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

            # Decode block identity for this time window
            decode_result = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                                   pre_time=PRE_TIME, post_time=POST_TIME,
                                   classifier=DECODER, cross_validation=VALIDATION,
                                   num_splits=NUM_SPLITS)

            # Add to dataframe
            nickname = ses_info[i]['subject']
            ses_date = ses_info[i]['start_time'][:10]
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                                 'f1': decode_result['f1'],
                                 'auroc': decode_result['auroc'],
                                 'ML': probes.trajectory[p]['x'],
                                 'AP': probes.trajectory[p]['y'],
                                 'DV': probes.trajectory[p]['z'],
                                 'phi': probes.trajectory[p]['phi'],
                                 'theta': probes.trajectory[p]['theta'],
                                 'depth': probes.trajectory[p]['depth']}))

decoding_result.to_csv(join(DATA_PATH, 'decoding_block_all_cortex'))

# %% Plot
Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=decoding_result.sort_values(by='f1'), hue='f1',
                         palette='RdBu_r', s=100, hue_norm=(0.3, 0.7), ax=ax1)

# Fix legend
leg = plot_h.legend(loc=(0.75, 0.5))
leg.texts[0].set_text('F1')

plot_settings()
plt.savefig(join(FIG_PATH, 'block_decode_all_cortex'))
