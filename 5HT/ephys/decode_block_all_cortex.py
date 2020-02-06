#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

@author: guido
"""

from os import listdir
from os.path import join, isfile
import alf.io as ioalf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
PRE_TIME = 0.5
POST_TIME = 0
DECODER = 'bayes'  # bayes, regression or forest
ITERATIONS = 500
NUM_SPLITS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Blocks')

# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times', details=True)

decoding_result = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    session_path = one_session_path(eid)
    spikes = one.load_object(eid, 'spikes', download_only=True)
    if len(spikes) != 0:
        probes = one.load_object(eid, 'probes', download_only=False)
        for p in range(len(probes['trajectory'])):
            if probes['trajectory'][p]['theta'] == 15:
                probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
                try:
                    spikes = alf.io.load_object(probe_path, object='spikes')
                    clusters = alf.io.load_object(probe_path, object='clusters')
                except Exception:
                    continue
                trials = one.load_object(eid, 'trials')

                # Only use single units
                spikes.times = spikes.times[np.isin(
                        spikes.clusters, clusters.metrics.cluster_id[
                                            clusters.metrics.ks2_label == 'good'])]
                spikes.clusters = spikes.clusters[np.isin(
                        spikes.clusters, clusters.metrics.cluster_id[
                                            clusters.metrics.ks2_label == 'good'])]
                clusters.depths = clusters.depths[clusters.metrics.ks2_label == 'good']
                cluster_ids = clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good']

                # Only use units from cortex
                spikes.times = spikes.times[np.isin(
                        spikes.clusters, clusters.metrics.cluster_id[clusters.depths < 1000])]
                spikes.clusters = spikes.clusters[np.isin(
                        spikes.clusters, clusters.metrics.cluster_id[clusters.depths < 1000])]

                # Get trial vectors
                trial_times = trials.goCue_times[((trials.probabilityLeft > 0.55)
                                                  | (trials.probabilityLeft < 0.55))]
                trial_blocks = (trials.probabilityLeft[
                                    ((trials.probabilityLeft > 0.55)
                                     | (trials.probabilityLeft < 0.55))] > 0.55).astype(int)
                trial_blocks_shuffle = trial_blocks.copy()

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
                f1_scores = np.empty(ITERATIONS)
                f1_scores_shuffle = np.empty(ITERATIONS)
                for it in range(ITERATIONS):
                    f1_scores[it], _ = decoding(resp, trial_blocks, clf, NUM_SPLITS)
                    np.random.shuffle(trial_blocks_shuffle)
                    f1_scores_shuffle[it], _ = decoding(resp, trial_blocks_shuffle,
                                                        clf, NUM_SPLITS)
                f1_over_shuffled = np.mean(f1_scores) - np.mean(f1_scores_shuffle)

                # Add to dataframe
                nickname = ses_info[i]['subject']
                ses_date = ses_info[i]['start_time'][:10]
                decoding_result = decoding_result.append(pd.DataFrame(
                    index=[0], data={'subject': nickname, 'date': ses_date, 'eid': eid,
                                     'f1_over_shuffled': f1_over_shuffled,
                                     'ML': probes.trajectory[p]['x'],
                                     'AP': probes.trajectory[p]['y'],
                                     'DV': probes.trajectory[p]['z'],
                                     'phi': probes.trajectory[p]['phi'],
                                     'theta': probes.trajectory[p]['theta'],
                                     'depth': probes.trajectory[p]['depth']}))
