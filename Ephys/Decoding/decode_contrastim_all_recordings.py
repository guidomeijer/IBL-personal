#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Decode whether a stimulus is consistent or inconsistent with the block for all recordings

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
PRE_TIME = 0
POST_TIME = 0.3
MIN_CONTRAST = 0.1
MIN_TRIALS = 500
MIN_NEURONS = 20
NUM_SPLITS = 1

DECODER = 'forest'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times',
                            task_protocol='_iblrig_tasks_ephysChoiceWorld', details=True)

# Initialize decoder
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

results = pd.DataFrame()
for s, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (s+1, len(eids)))
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
        if (probes['trajectory'][p]['theta'] == 15) and (probes['trajectory'][p]['depth'] < 4500):
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                print('Could not load spikes or clusters Bunch, skipping recording')
                continue
            if trials.stimOn_times.shape[0] < MIN_TRIALS:
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

            # Get matrix of neuronal responses
            times = np.column_stack(((trials.stimOn_times - PRE_TIME),
                                     (trials.stimOn_times + POST_TIME)))
            resp, cluster_ids = bb.task._get_spike_counts_in_bins(spikes.times, spikes.clusters,
                                                                  times)
            resp = np.rot90(resp)

            # Get trial indices of inconsistent trials during left high blocks
            incon_l_block = ((trials.probabilityLeft > 0.55)
                             & (trials.contrastRight > MIN_CONTRAST))
            cons_l_block = ((trials.probabilityLeft > 0.55)
                            & (trials.contrastLeft > MIN_CONTRAST))
            consistent_l = np.zeros(cons_l_block.shape[0])
            consistent_l[cons_l_block == 1] = 1
            consistent_l[incon_l_block == 1] = 2
            resp_l = resp[(consistent_l == 1) | (consistent_l == 2), :]
            consistent_l = consistent_l[(consistent_l == 1) | (consistent_l == 2)]

            # Abort if there are more inconsisten than consistent trials (something is wrong)
            if np.sum(consistent_l == 2) > np.sum(consistent_l == 1):
                continue

            # Decode whether stimulus on the left is consistent or inconsistent with block prob.
            f1_l, auroc_l, _ = decoding(resp_l, consistent_l, clf, NUM_SPLITS)

            # Get trial indices of inconsistent trials during right high blocks
            incon_r_block = ((trials.probabilityLeft < 0.45)
                             & (trials.contrastLeft > MIN_CONTRAST))
            cons_r_block = ((trials.probabilityLeft < 0.45)
                            & (trials.contrastRight > MIN_CONTRAST))
            right_times = trials.stimOn_times[(cons_r_block == 1) | (incon_r_block == 1)]
            consistent_r = np.zeros(cons_r_block.shape[0])
            consistent_r[cons_r_block == 1] = 1
            consistent_r[incon_r_block == 1] = 2
            resp_r = resp[(consistent_r == 1) | (consistent_r == 2), :]
            consistent_r = consistent_r[(consistent_r == 1) | (consistent_r == 2)]

            # Abort if there are more inconsisten than consistent trials (something is wrong)
            if np.sum(consistent_r == 2) > np.sum(consistent_r == 1):
                continue

            # Decode whether stimulus on the left is consistent or inconsistent with block prob.
            f1_r, auroc_r, _ = decoding(resp_r, consistent_r, clf, NUM_SPLITS)

            # Add to dataframe
            nickname = ses_info[s]['subject']
            ses_date = ses_info[s]['start_time'][:10]
            results = results.append(pd.DataFrame(
                index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                                 'f1_l': f1_l, 'f1_r': f1_r,
                                 'auroc_l': auroc_l, 'auroc_r': auroc_r,
                                 'ML': probes.trajectory[p]['x'],
                                 'AP': probes.trajectory[p]['y'],
                                 'DV': probes.trajectory[p]['z'],
                                 'phi': probes.trajectory[p]['phi'],
                                 'theta': probes.trajectory[p]['theta'],
                                 'depth': probes.trajectory[p]['depth']}))

results.to_csv(join(SAVE_PATH, 'decode_contrastim_all_recordings.csv'))

# Plot
Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=results, hue='auroc_r',
                         palette='YlOrRd', s=100, ax=ax1)

# Fix legend
leg = plot_h.legend(loc=(0.8, 0.5))
leg.texts[0].set_text('Decoding perf.')

plot_settings()
plt.savefig(join(FIG_PATH, 'decode_contrastim_all_recordings'))
