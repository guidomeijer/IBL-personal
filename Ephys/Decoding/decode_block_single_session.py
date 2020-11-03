#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import scipy
from brainbox.population import decode, _get_spike_counts_in_bins
import pandas as pd
from scipy.stats import wilcoxon
import seaborn as sns
import alf
from ephys_functions import (paths, figure_style, sessions_with_hist, check_trials,
                             combine_layers_cortex)
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

EID = '510b1a50-825d-44ce-86f6-9678f5396e02'
REGION = 'CM'
PROBE = 'probe00'
PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'bayes'
VALIDATION = 'kfold'
NUM_SPLITS = 5
DOWNLOAD_TRIALS = False
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')


# %%
# Load in data
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(EID, one=one)
ses_path = one.path_from_eid(EID)
if DOWNLOAD_TRIALS:
    _ = one.load(EID, dataset_types=['trials.stimOn_times', 'trials.probabilityLeft'],
                 download_only=True, clobber=True)
trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

# Get trial vectors
incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
trial_times = trials.stimOn_times[incl_trials]
probability_left = trials.probabilityLeft[incl_trials]
trial_blocks = (trials.probabilityLeft[incl_trials] == 0.8).astype(int)

# Get clusters in this brain region
region_clusters = combine_layers_cortex(clusters[PROBE]['acronym'])
clusters_in_region = clusters[PROBE].metrics.cluster_id[region_clusters == REGION]

# Select spikes and clusters
spks_region = spikes[PROBE].times[np.isin(spikes[PROBE].clusters, clusters_in_region)]
clus_region = spikes[PROBE].clusters[np.isin(spikes[PROBE].clusters,
                                             clusters_in_region)]

# Decode block identity
decode_5fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME,
                      classifier=DECODER, cross_validation='kfold',
                      num_splits=5)

shuffle_5fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME,
                       classifier=DECODER, cross_validation='kfold',
                       num_splits=5, shuffle=True)

phase_5fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='kfold',
                     num_splits=5, phase_rand=True)

decode_2fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME,
                      classifier=DECODER, cross_validation='kfold',
                      num_splits=2)

shuffle_2fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME,
                       classifier=DECODER, cross_validation='kfold',
                       num_splits=2, shuffle=True)

phase_2fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='kfold',
                     num_splits=2, phase_rand=True)

decode_loe = decode(spks_region, clus_region, trial_times, trial_blocks,
                    pre_time=PRE_TIME, post_time=POST_TIME,
                    classifier=DECODER, cross_validation='leave-one-out')

shuffle_loe = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='leave-one-out', shuffle=True)

phase_loe = decode(spks_region, clus_region, trial_times, trial_blocks,
                   pre_time=PRE_TIME, post_time=POST_TIME,
                   classifier=DECODER, cross_validation='leave-one-out', phase_rand=True)

decode_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME, prob_left=probability_left,
                      classifier=DECODER, cross_validation='block')

shuffle_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME, prob_left=probability_left,
                       classifier=DECODER, cross_validation='block', shuffle=True)

phase_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME, prob_left=probability_left,
                     classifier=DECODER, cross_validation='block', phase_rand=True)

"""
# Get matrix of all neuronal responses
times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
pop_vector, cluster_ids = _get_spike_counts_in_bins(spks_region, clus_region, times)
pop_vector = np.rot90(pop_vector)

# Phase randomization
rand_pop_vector = np.empty(pop_vector.shape)
frequencies = int((pop_vector.shape[0] - 1) / 2)
fsignal = scipy.fft.fft(pop_vector, axis=0)
power = np.abs(fsignal[1:1+frequencies])
phases = 2*np.pi*np.random.rand(frequencies)
for i in range(pop_vector.shape[1]):
    newfsignal = fsignal[0, i]
    newfsignal = np.append(newfsignal, np.exp(1j * phases) * power[:, i])
    newfsignal = np.append(newfsignal, np.flip(np.exp(-1j * phases) * power[:, i]))
    newsignal = scipy.fft.ifft(newfsignal)
    rand_pop_vector[:, i] = np.abs(newsignal.real)
"""

# %%
f, ax1 = plt.subplots(1, 1)
ax1.bar(np.arange(13), [decode_5fold['accuracy'], shuffle_5fold['accuracy'], phase_5fold['accuracy'],
                        decode_2fold['accuracy'], shuffle_2fold['accuracy'], phase_2fold['accuracy'],
                        decode_loe['accuracy'], shuffle_loe['accuracy'], phase_loe['accuracy'],
                        decode_block['accuracy'], shuffle_block['accuracy'], phase_block['accuracy'],
                        np.sum(trial_blocks == 0) / len(trial_blocks)])
ax1.set(xticks=np.arange(13),
        xticklabels=['5fold', 'shuffle', 'phase', '2fold', 'shuffle', 'phase', 'loe', 'shuffle',
                     'phase', 'block', 'shuffle',  'phase', 'chance'],
        ylabel='Decoding accuracy', title='Central medial nucleus of the thalamus')


# ax1.plot(np.arange(len(decode_5fold['probabilities'][0])), decode_5fold['probabilities'][0])
