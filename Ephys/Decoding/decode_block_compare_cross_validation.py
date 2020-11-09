#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import decode
import alf
from ephys_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

EID = '510b1a50-825d-44ce-86f6-9678f5396e02'
REGION = 'CM'  # Central medial nucleus of the thalamus
PROBE = 'probe00'
PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'bayes'
ITERATIONS = 1000
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
                       num_splits=5, shuffle=True, iterations=ITERATIONS)

phase_5fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='kfold',
                     num_splits=5, phase_rand=True, iterations=ITERATIONS)

decode_2fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME,
                      classifier=DECODER, cross_validation='kfold',
                      num_splits=2)

shuffle_2fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME,
                       classifier=DECODER, cross_validation='kfold',
                       num_splits=2, shuffle=True, iterations=ITERATIONS)

phase_2fold = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='kfold',
                     num_splits=2, phase_rand=True, iterations=ITERATIONS)

decode_loe = decode(spks_region, clus_region, trial_times, trial_blocks,
                    pre_time=PRE_TIME, post_time=POST_TIME,
                    classifier=DECODER, cross_validation='leave-one-out')

shuffle_loe = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='leave-one-out',
                     shuffle=True, iterations=ITERATIONS)

phase_loe = decode(spks_region, clus_region, trial_times, trial_blocks,
                   pre_time=PRE_TIME, post_time=POST_TIME,
                   classifier=DECODER, cross_validation='leave-one-out',
                   phase_rand=True, iterations=ITERATIONS)

decode_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME, prob_left=probability_left,
                      classifier=DECODER, cross_validation='block')

shuffle_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME, prob_left=probability_left,
                       classifier=DECODER, cross_validation='block', shuffle=True,
                       iterations=ITERATIONS)

phase_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME, prob_left=probability_left,
                     classifier=DECODER, cross_validation='block', phase_rand=True,
                     iterations=ITERATIONS)

# %%
figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(10,10), dpi=150)
ax1.bar(np.arange(15), [decode_5fold['accuracy'], shuffle_5fold['accuracy'].mean(),
                        phase_5fold['accuracy'].mean(), np.nan, decode_2fold['accuracy'],
                        shuffle_2fold['accuracy'].mean(), phase_2fold['accuracy'].mean(), np.nan,
                        decode_loe['accuracy'], shuffle_loe['accuracy'].mean(),
                        phase_loe['accuracy'].mean(), np.nan, decode_block['accuracy'],
                        shuffle_block['accuracy'].mean(), phase_block['accuracy'].mean()],
        yerr=[0, shuffle_5fold['accuracy'].std(), phase_5fold['accuracy'].std(), 0,
              0, shuffle_2fold['accuracy'].std(), phase_2fold['accuracy'].std(), 0,
              0, shuffle_loe['accuracy'].std(), phase_loe['accuracy'].std(), 0,
              0, shuffle_block['accuracy'].std(), phase_block['accuracy'].std()])
ax1.set(xticks=np.arange(15),
        xticklabels=['5fold', 'shuffle', 'phase', '', '2fold', 'shuffle', 'phase', '',
                     'loe', 'shuffle', 'phase', '', 'block', 'shuffle',  'phase'],
        ylabel='Decoding accuracy', title='Central medial nucleus of the thalamus')

print(('p 5-fold shuffle: %.2f \np 5-fold phase: %.2f \np 2-fold shuffle: %.2f \np 2-fold phase: %.2f\n'
      + 'p leave-one-out shuffle: %.2f\np leave-one-out phase: %.2f\np block shuffle: %.2f'
      + '\np block phase: %.2f') % (
          (np.sum(shuffle_5fold['accuracy'] > decode_5fold['accuracy'])
           / shuffle_5fold['accuracy'].shape[0]),
          (np.sum(phase_5fold['accuracy'] > decode_5fold['accuracy'])
           / phase_5fold['accuracy'].shape[0]),
          (np.sum(shuffle_2fold['accuracy'] > decode_2fold['accuracy'])
           / shuffle_2fold['accuracy'].shape[0]),
          (np.sum(phase_2fold['accuracy'] > decode_2fold['accuracy'])
           / phase_2fold['accuracy'].shape[0]),
          (np.sum(shuffle_loe['accuracy'] > decode_loe['accuracy'])
           / shuffle_loe['accuracy'].shape[0]),
          (np.sum(phase_loe['accuracy'] > decode_loe['accuracy'])
           / phase_loe['accuracy'].shape[0]),
          (np.sum(shuffle_block['accuracy'] > decode_block['accuracy'])
           / shuffle_block['accuracy'].shape[0]),
          (np.sum(phase_block['accuracy'] > decode_block['accuracy'])
           / phase_block['accuracy'].shape[0])
          ))
# ax1.plot(np.arange(len(decode_5fold['probabilities'][0])), decode_5fold['probabilities'][0])
