# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

Decode whether a stimulus is consistent or inconsistent with the block for frontal and control
recordings seperated by probe depth.

@author: guido
"""

from os import listdir
from os.path import join
import alf.io as ioalf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import paths, sessions, decoding, plot_settings, get_spike_counts_in_bins

# Settings
LAB = 'danlab'
SUBJECT = 'DY_011'
DATE = '2020-01-30'
PROBE = '00'
DEPTH_BIN_CENTERS = np.arange(200, 4000, 200)
DEPTH_BIN_SIZE = 300
PRE_TIME = 0
POST_TIME = 0.3
MIN_CONTRAST = 0.1
DECODER = 'forest'  # bayes, regression or forest
NUM_SPLITS = 3

DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'ContraStim')

# Get paths
ses_nr = listdir(join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE))[0]
session_path = join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE, ses_nr)
alf_path = join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE, ses_nr, 'alf')
probe_path = join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE, ses_nr, 'alf', 'probe%s' % PROBE)

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
f1_score = np.empty(len(DEPTH_BIN_CENTERS))
auroc = np.empty(len(DEPTH_BIN_CENTERS))
n_clusters = np.empty(len(DEPTH_BIN_CENTERS))
for j, depth in enumerate(DEPTH_BIN_CENTERS):
    print('Decoding block identity from depth %d..' % depth)
    depth_clusters = cluster_ids[((clusters.depths > depth-(DEPTH_BIN_SIZE/2))
                                  & (clusters.depths < depth+(DEPTH_BIN_SIZE/2)))]
    if len(depth_clusters) <= 2:
        n_clusters[j] = len(depth_clusters)
        f1_score[j] = np.nan
        auroc[j] = np.nan
        continue
    f1_score[j], auroc[j], _ = decoding(resp[:, np.isin(cluster_ids, depth_clusters)],
                                        trial_consistent, clf, NUM_SPLITS)
    n_clusters[j] = len(depth_clusters)

# Plot decoding versus depth
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
ax1.plot(f1_score, DEPTH_BIN_CENTERS, lw=2)
ax1.set(ylabel='Depth (um)', xlabel='Decoding performance (AUROC)',
        title='Decoding whether stimulus side is consistent with block probability',
        xlim=[-0.1, 0.4])

ax2.plot(n_clusters, DEPTH_BIN_CENTERS, lw=2)
ax2.set(xlabel='Number of neurons')
ax2.invert_yaxis()
plot_settings()
plt.savefig(join(FIG_PATH, '%s_%s_%s' % (DECODER,
                                            sessions.loc[i, 'subject'],
                                            sessions.loc[i, 'date'])))
plt.close(f)

# Save decoding performance
np.save(join(SAVE_PATH, 'Decoding', 'ContraStim',
             '%s_%s_%s' % (DECODER, sessions.loc[i, 'subject'],
                              sessions.loc[i, 'date'])), f1_over_shuffled)
