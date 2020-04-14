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
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import paths, decoding, get_spike_counts_in_bins

# Session
LAB = 'danlab'
SUBJECT = 'DY_011'
DATE = '2020-01-30'
PROBE = '00'

# Settings
PRE_TIME = 0.2
POST_TIME = 0.3
MIN_CONTRAST = 0.05
DECODER = 'regression'  # bayes, regression or forest
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
"""
inconsistent = (((trials.probabilityLeft > 0.55)
                 & (trials.contrastRight > MIN_CONTRAST)) |
                ((trials.probabilityLeft < 0.45)
                 & (trials.contrastLeft > MIN_CONTRAST)))
consistent = (((trials.probabilityLeft < 0.45)
               & (trials.contrastRight > MIN_CONTRAST)) |
              ((trials.probabilityLeft > 0.55)
               & (trials.contrastRight > MIN_CONTRAST)))
"""
inconsistent = (((trials.probabilityLeft > 0.55)
                 & (trials.contrastRight > MIN_CONTRAST)))
consistent = (((trials.probabilityLeft < 0.45)
               & (trials.contrastRight > MIN_CONTRAST)))
"""
inconsistent = (((trials.probabilityLeft < 0.45)
                 & (trials.contrastLeft > MIN_CONTRAST)))
consistent = (((trials.probabilityLeft > 0.55)
               & (trials.contrastLeft > MIN_CONTRAST)))
"""

trial_times = trials.stimOn_times[(consistent == 1) | (inconsistent == 1)]
trial_consistent = np.zeros(consistent.shape[0])
trial_consistent[consistent == 1] = 1
trial_consistent[inconsistent == 1] = 2
trial_consistent = trial_consistent[(consistent == 1) | (inconsistent == 1)]

# Get matrix of all neuronal responses
times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
resp, cluster_ids = get_spike_counts_in_bins(spikes.times, spikes.clusters, times)
resp = np.rot90(resp)

# Decode
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')
f1, auroc, _ = decoding(resp, trial_consistent, clf, NUM_SPLITS)
print('F1 score: %.2f, AUROC: %.2f' % (np.round(f1, 2), np.round(auroc, 2)))
