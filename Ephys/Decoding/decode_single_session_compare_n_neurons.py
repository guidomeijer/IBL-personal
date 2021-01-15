#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.population import decode
import seaborn as sns
import alf
from my_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

"""
TARGET = 'stim-side'
EID = 'aad23144-0e52-4eac-80c5-c4ee2decb198'
REGION = 'VISa'
PROBE = 'probe00'


TARGET = 'reward'
EID = '0cbeae00-e229-4b7d-bdcc-1b0569d7e0c3'
REGION = 'CP'
PROBE = 'probe01'


TARGET = 'choice'
EID = 'ff4187b5-4176-4e39-8894-53a24b7cf36b'
REGION = 'MOs'
PROBE = 'probe00'

"""
TARGET = 'block'
EID = 'b658bc7d-07cd-4203-8a25-7b16b549851b'
REGION = 'CP'
PROBE = 'probe01'

N_NEURONS = np.arange(10, 151, 20)
N_NEURON_PICK = 100
MIN_CONTRAST = 0.1
DECODER = 'bayes-multinomial'
VALIDATION = 'kfold-interleaved'
CHANCE_LEVEL = 'shuffle'
NUM_SPLITS = 5
ITERATIONS = 100
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Sessions', DECODER, 'n_neurons')

# Time windows
if (TARGET == 'block') | (TARGET == 'block-first') | (TARGET == 'block-last'):
    PRE_TIME = 0.6
    POST_TIME = 0.1
elif TARGET == 'block-stim':
    PRE_TIME = 0
    POST_TIME = 0.3
elif TARGET == 'stim-side':
    MIN_CONTRAST = 0.2
    PRE_TIME = 0
    POST_TIME = 0.3
elif TARGET == 'reward':
    PRE_TIME = 0
    POST_TIME = 0.5
elif TARGET == 'choice':
    PRE_TIME = 0.2
    POST_TIME = 0

# %%
# Load in data
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(EID, aligned=True, one=one)
ses_path = one.path_from_eid(EID)
trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

# Get trial vectors based on decoding target
if TARGET == 'block':
    incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_ids = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)
elif TARGET == 'stim-side':
    incl_trials = (((trials.contrastLeft > MIN_CONTRAST)
                    | (trials.contrastRight > MIN_CONTRAST))
                   & (trials.probabilityLeft != 0.5))
    trial_times = trials.stimOn_times[incl_trials]
    trial_ids = np.isnan(trials.contrastLeft[incl_trials]).astype(int)
elif TARGET == 'reward':
    incl_trials = (trials.choice != 0) & (trials.probabilityLeft != 0.5)
    trial_times = trials.feedback_times[incl_trials]
    trial_ids = (trials.feedbackType[incl_trials] == -1).astype(int)
elif TARGET == 'choice':
    incl_trials = (((trials.choice != 0) & (~np.isnan(trials.feedback_times)))
                   & (trials.probabilityLeft != 0.5))
    trial_times = trials.feedback_times[incl_trials]
    trial_ids = (trials.choice[incl_trials] == 1).astype(int)
    choice_ratio = ((np.sum(trial_ids == 0) - np.sum(trial_ids == 1))
                    / (np.sum(trial_ids == 0) + np.sum(trial_ids == 1)))

# Get clusters in this brain region
region_clusters = combine_layers_cortex(clusters[PROBE]['acronym'])
clusters_in_region = clusters[PROBE].metrics.cluster_id[region_clusters == REGION]

# Select spikes and clusters
spks_region = spikes[PROBE].times[np.isin(spikes[PROBE].clusters, clusters_in_region)]
clus_region = spikes[PROBE].clusters[np.isin(spikes[PROBE].clusters,
                                             clusters_in_region)]

decode_subselects = pd.DataFrame()
for i, n_neurons in enumerate(N_NEURONS):
    print('Decoding from groups of %d neurons [%d of %d]' % (n_neurons, i + 1, len(N_NEURONS)))
    for j in range(N_NEURON_PICK):

        # Subselect neurons
        use_neurons = np.random.choice(clusters_in_region, n_neurons, replace=False)

        # Decode
        decode_result = decode(spks_region[np.isin(clus_region, use_neurons)],
                               clus_region[np.isin(clus_region, use_neurons)],
                               trial_times, trial_ids,
                               pre_time=PRE_TIME, post_time=POST_TIME,
                               classifier=DECODER, cross_validation=VALIDATION,
                               num_splits=NUM_SPLITS)
        decode_subselects = decode_subselects.append(pd.DataFrame(
                    index=[decode_subselects.shape[0] + 1], data={
                        'accuracy': decode_result['accuracy'], 'null': 'Original',
                        'n_neurons': n_neurons}))
        if CHANCE_LEVEL == 'shuffle':
            decode_chance = decode(spks_region[np.isin(clus_region, use_neurons)],
                                   clus_region[np.isin(clus_region, use_neurons)],
                                   trial_times, trial_ids,
                                   pre_time=PRE_TIME, post_time=POST_TIME,
                                   classifier=DECODER, cross_validation=VALIDATION,
                                   num_splits=NUM_SPLITS, shuffle=True, iterations=ITERATIONS)
        elif CHANCE_LEVEL == 'pseudo-blocks':
            decode_chance = decode(spks_region[np.isin(clus_region, use_neurons)],
                                   clus_region[np.isin(clus_region, use_neurons)],
                                   trial_times, trial_ids,
                                   pre_time=PRE_TIME, post_time=POST_TIME,
                                   classifier=DECODER, cross_validation=VALIDATION,
                                   num_splits=NUM_SPLITS, pseudo_blocks=True,
                                   iterations=ITERATIONS)
        decode_subselects = decode_subselects.append(pd.DataFrame(
                    index=[decode_subselects.shape[0] + 1], data={
                        'accuracy': decode_chance['accuracy'].mean(), 'null': 'Shuffled',
                        'n_neurons': n_neurons}))

# %%
figure_style()
f, ax1 = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
sns.lineplot(x='n_neurons', y='accuracy', hue='null', data=decode_subselects)
ax1.set(xlabel='Number of neurons', ylabel='Decoding accuracy (%)')
if TARGET == 'stim-side':
    ax1.set(title='Decoding of stimulus side, region: %s' % REGION)
elif TARGET == 'block':
    ax1.set(title='Decoding of stimulus prior, region: %s' % REGION)
elif TARGET == 'reward':
    ax1.set(title='Decoding of reward, region: %s' % REGION)
elif TARGET == 'choice':
    ax1.set(title='Decoding of motor response, region: %s' % REGION)
legend = ax1.legend(frameon=False)

plt.savefig(join(FIG_PATH, '%s_%s' % (REGION, TARGET)))
