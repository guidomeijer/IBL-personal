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
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import paths, decoding, plot_settings, get_spike_counts_in_bins

# Session
LAB = 'danlab'
SUBJECT = 'DY_011'
DATE = '2020-01-30'
PROBE = '00'

# Settings
N_NEURONS = 30
RANDOM_PICKS = 500
SUP_DEEP = 1500
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

# Decode contrastim superficial
auroc_sup = np.empty(RANDOM_PICKS)
print('Decoding block identity from superficial clusters..')
depth_clusters = cluster_ids[clusters.depths < SUP_DEEP]
for i in range(RANDOM_PICKS):
    if np.mod(i, 50) == 0:
        print('Random pick %d of %d' % (i, RANDOM_PICKS))
    use_clusters = np.random.choice(depth_clusters, size=N_NEURONS, replace=False)
    _, auroc_sup[i], _ = decoding(resp[:, np.isin(cluster_ids, use_clusters)],
                                  trial_consistent, clf, NUM_SPLITS)

# Decode contrastim superficial
auroc_deep = np.empty(RANDOM_PICKS)
print('Decoding block identity from deep clusters..')
depth_clusters = cluster_ids[clusters.depths > SUP_DEEP]
for i in range(RANDOM_PICKS):
    if np.mod(i, 50) == 0:
        print('Random pick %d of %d' % (i, RANDOM_PICKS))
    use_clusters = np.random.choice(depth_clusters, size=N_NEURONS, replace=False)
    _, auroc_deep[i], _ = decoding(resp[:, np.isin(cluster_ids, use_clusters)],
                                   trial_consistent, clf, NUM_SPLITS)

# Put data in dataframe and save
results = pd.DataFrame(data={'subject': SUBJECT, 'date': DATE, 'probe': PROBE,
                             'auroc': np.append(auroc_sup, auroc_deep),
                             'sup_deep': ['Superficial']*RANDOM_PICKS + ['Deep']*RANDOM_PICKS})

# %% Plot

f, ax1 = plt.subplots(1, 1, figsize=(8, 8))
sns.boxplot(x='sup_deep', y='auroc', data=results, ax=ax1)

