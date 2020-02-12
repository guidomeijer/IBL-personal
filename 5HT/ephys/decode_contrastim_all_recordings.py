#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import alf.io
import seaborn as sns
import brainbox as bb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import paths, plot_settings, one_session_path, decoding
from oneibl.one import ONE
one = ONE()

# Settings
PRE_TIME = 1
POST_TIME = -0.5
MIN_CONTRAST = 0.1
MIN_TRIALS = 200
MIN_NEURONS = 20
ITERATIONS = 500
NUM_SPLITS = 5
DECODER = 'forest'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times', details=True)

results = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
    probes = one.load_object(eid, 'probes', download_only=False)
    if (not hasattr(trials, 'stimOn_times')
            or (trials.stimOn_times.shape[0] != trials.probabilityLeft.shape[0])
            or (not hasattr(probes, 'trajectory'))):
        print('Invalid data, skipping recording')
        continue
    for p in range(len(probes['trajectory'])):
        # Select shallow penetrations
        # if (probes['trajectory'][p]['theta'] == 15)and (probes['trajectory'][p]['depth'] < 4500):
        probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
        try:
            spikes = alf.io.load_object(probe_path, object='spikes')
            clusters = alf.io.load_object(probe_path, object='clusters')
        except Exception:
            print('Could not load spikes or clusters Bunch, skipping recording')
            continue

        # Only use good single units
        clusters_to_use = clusters.metrics.ks2_label == 'good'
        if clusters_to_use.sum() < MIN_NEURONS:
            continue
        spikes.times = spikes.times[
                np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
        spikes.clusters = spikes.clusters[
                np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
        clusters.depths = clusters.depths[clusters_to_use]
        cluster_ids = clusters.metrics.cluster_id[clusters_to_use]

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
        if trial_times.shape[0] < MIN_TRIALS:
            continue
        trial_consistent = np.zeros(consistent.shape[0])
        trial_consistent[consistent == 1] = 1
        trial_consistent[inconsistent == 1] = 2
        trial_consistent = trial_consistent[(consistent == 1) | (inconsistent == 1)]
        trial_consistent_shuffle = trial_consistent.copy()

        # Get matrix of all neuronal responses
        times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
        resp, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times, spikes.clusters, times)
        resp = np.rot90(resp)

        # Initialize decoder
        if DECODER == 'forest':
            clf = RandomForestClassifier(n_estimators=100)
        elif DECODER == 'bayes':
            clf = GaussianNB()
        elif DECODER == 'regression':
            clf = LogisticRegression(solver='liblinear', multi_class='auto')
        else:
            raise Exception('DECODER must be forest, bayes or regression')

        # Decode block identity
        aucroc_it = np.empty(ITERATIONS)
        f1_it = np.empty(ITERATIONS)
        for it in range(ITERATIONS):
            _, f1_it[it], aucroc_it[it], _ = decoding(resp, trial_consistent, clf, NUM_SPLITS)
        aucroc = np.mean(aucroc_it)
        f1 = np.mean(f1_it)

        # Add to dataframe
        nickname = ses_info[i]['subject']
        ses_date = ses_info[i]['start_time'][:10]
        results = results.append(pd.DataFrame(
            index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                             'f1': f1, 'aucroc': aucroc,
                             'ML': probes.trajectory[p]['x'],
                             'AP': probes.trajectory[p]['y'],
                             'DV': probes.trajectory[p]['z'],
                             'phi': probes.trajectory[p]['phi'],
                             'theta': probes.trajectory[p]['theta'],
                             'depth': probes.trajectory[p]['depth']}))

results.to_csv(join(DATA_PATH, 'decode_contrastim_all_recordings'))

# Plot
Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=results, hue='aucroc', palette='YlOrRd',
                         hue_norm=(0.5, 0.6), s=100, ax=ax1)

# Fix legend
leg = plot_h.legend(loc=(1.05, 0.5))
leg.texts[0].set_text('auROC')
leg.texts[1].set_text('0.5')
leg.texts[2].set_text('0.54')
leg.texts[3].set_text('0.57')
leg.texts[4].set_text('0.6')


plot_settings()
plt.savefig(join(FIG_PATH, 'decode_contrastim_all_recordings'))
