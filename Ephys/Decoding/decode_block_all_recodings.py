#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Decode left/right block identity from all recordings

@author: guido
"""

from os import listdir
from os.path import join, isfile
import alf.io as ioalf
import matplotlib.pyplot as plt
import brainbox as bb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from functions_5HT import (download_data, paths, sessions, decoding, plot_settings,
                           get_spike_counts_in_bins)

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
eids, ses_info = one.search(dataset_types='spikes.times',
                            task_protocol='_iblrig_tasks_ephysChoiceWorld', details=True)

decode_results = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    session_path = one_session_path(eid)
    spikes = one.load_object(eid, 'spikes', download_only=True)
    if len(spikes) != 0:
        probes = one.load_object(eid, 'probes', download_only=False)
        for p in range(len(probes['trajectory'])):
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                continue
            trials = one.load_object(eid, 'trials')


    # Download data if required
    if DOWNLOAD is True:
        download_data(sessions.loc[i, 'subject'], sessions.loc[i, 'date'])

    # Get paths
    ses_nr = listdir(join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                          sessions.loc[i, 'subject'], sessions.loc[i, 'date']))[0]
    session_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects',
                        sessions.loc[i, 'subject'], sessions.loc[i, 'date'], ses_nr)
    alf_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                    sessions.loc[i, 'date'], ses_nr, 'alf')
    probe_path = join(DATA_PATH, sessions.loc[i, 'lab'], 'Subjects', sessions.loc[i, 'subject'],
                      sessions.loc[i, 'date'], ses_nr, 'alf', 'probe%s' % sessions.loc[i, 'probe'])

    if not isfile(join(
            SAVE_PATH, 'Decoding', 'Blocks', '%s_%s_%s_%s.npy' % (FRONTAL_CONTROL, DECODER,
                                                                  sessions.loc[i, 'subject'],
                                                                  sessions.loc[i, 'date']))) or (
                                                                          OVERWRITE is True):
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
        trial_times = trials.goCue_times[((trials.probabilityLeft > 0.45)
                                          | (trials.probabilityLeft < 0.45))]
        trial_blocks = (trials.probabilityLeft[((trials.probabilityLeft > 0.55)
                                                | (trials.probabilityLeft < 0.45))] > 0.55).astype(
                                                                                            int)
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
        f1_over_shuffled = np.empty(len(DEPTH_BIN_CENTERS))
        n_clusters = np.empty(len(DEPTH_BIN_CENTERS))
        significant_depth = np.zeros(len(DEPTH_BIN_CENTERS), dtype=bool)
        for j, depth in enumerate(DEPTH_BIN_CENTERS):
            print('Decoding block identity from depth %d..' % depth)
            depth_clusters = cluster_ids[((clusters.depths > depth-(DEPTH_BIN_SIZE/2))
                                          & (clusters.depths < depth+(DEPTH_BIN_SIZE/2)))]
            if len(depth_clusters) <= 2:
                n_clusters[j] = len(depth_clusters)
                f1_over_shuffled[j] = np.nan
                continue
            f1_scores = np.empty(ITERATIONS)
            f1_scores_shuffle = np.empty(ITERATIONS)
            for it in range(ITERATIONS):
                f1_scores[it], _ = decoding(resp[:, np.isin(cluster_ids, depth_clusters)],
                                            trial_blocks, clf, NUM_SPLITS)
                np.random.shuffle(trial_blocks_shuffle)
                f1_scores_shuffle[it], _ = decoding(resp[:, np.isin(cluster_ids, depth_clusters)],
                                                    trial_blocks_shuffle, clf, NUM_SPLITS)
            f1_over_shuffled[j] = np.mean(f1_scores) - np.mean(f1_scores_shuffle)
            n_clusters[j] = len(depth_clusters)

            # Determine significance
            if np.percentile(f1_scores, 0.5) > np.mean(f1_scores_shuffle):
                significant_depth[j] = True

        # Plot decoding versus depth
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
        ax1.plot(f1_over_shuffled, DEPTH_BIN_CENTERS, lw=2)
        ax1.set(ylabel='Depth (um)', xlabel='Decoding performance\n(F1 score over shuffled)',
                title='Decoding of block identity', xlim=[-0.1, 0.4])
        # for j, (x, y) in enumerate(zip(f1_over_shuffled[significant_depth],
        #                                DEPTH_BIN_CENTERS[significant_depth])):
        #   ax1.text(x+0.02, y+30, '*', va='center')
        ax2.plot(n_clusters, DEPTH_BIN_CENTERS, lw=2)
        ax2.set(xlabel='Number of neurons')
        ax2.invert_yaxis()
        plot_settings()
        plt.savefig(join(FIG_PATH, '%s_%s_%s_%s' % (FRONTAL_CONTROL, DECODER,
                                                    sessions.loc[i, 'subject'],
                                                    sessions.loc[i, 'date'])))
        plt.close(f)

        # Save decoding performance
        np.save(join(SAVE_PATH, 'Decoding', 'Blocks',
                     '%s_%s_%s_%s' % (FRONTAL_CONTROL, DECODER, sessions.loc[i, 'subject'],
                                      sessions.loc[i, 'date'])), f1_over_shuffled)