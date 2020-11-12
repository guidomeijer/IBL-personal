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
from matplotlib.patches import Rectangle
import seaborn as sns
import alf
from ephys_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

EID = 'c9fec76e-7a20-4da4-93ad-04510a89473b'
REGION = 'ACAd'
PROBE = 'probe01'
PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'bayes'
ITERATIONS = 1000
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Single sessions', DECODER)

# %%
# Load in data
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(EID, aligned=True, one=one)
ses_path = one.path_from_eid(EID)
trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

# Get trial vectors
incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
trial_times = trials.stimOn_times[incl_trials]
probability_left = trials.probabilityLeft[incl_trials]
trial_blocks = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

# Get clusters in this brain region
region_clusters = combine_layers_cortex(clusters[PROBE]['acronym'])
clusters_in_region = clusters[PROBE].metrics.cluster_id[region_clusters == REGION]

# Select spikes and clusters
spks_region = spikes[PROBE].times[np.isin(spikes[PROBE].clusters, clusters_in_region)]
clus_region = spikes[PROBE].clusters[np.isin(spikes[PROBE].clusters,
                                             clusters_in_region)]

# Decode block identity
decode_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME,
                      classifier=DECODER, cross_validation='kfold',
                      num_splits=5)

phase_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier=DECODER, cross_validation='kfold',
                     num_splits=5, phase_rand=True, iterations=ITERATIONS)

shuffle_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME,
                       classifier=DECODER, cross_validation='kfold',
                       num_splits=5, shuffle=True, iterations=ITERATIONS)

# %%
figure_style()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10), dpi=150)
ax1.bar(np.arange(7), [decode_block['accuracy'], phase_block['accuracy'].mean(),
                        shuffle_block['accuracy'].mean(), np.nan, decode_block['f1'],
                        phase_block['f1'].mean(), shuffle_block['f1'].mean()],
        yerr=[0, phase_block['accuracy'].std(), shuffle_block['accuracy'].std(), 0,
              0, phase_block['f1'].std(), shuffle_block['f1'].std()])
ax1.set(xticks=np.arange(7),
        xticklabels=['acc.', 'phase', 'shuffle', '', 'f1', 'phase', 'shuffle'],
        ylabel='Decoding performance', title='Region: %s' % REGION)

block_colors = [sns.color_palette('colorblind', as_cmap=True)[0],
                sns.color_palette('colorblind', as_cmap=True)[3]]
block_trans = np.append([0], np.array(np.where(np.diff(trial_blocks) != 0)) + 1)
block_trans = np.append(block_trans, [trial_blocks.shape[0]])
for i, trans in enumerate(block_trans[:-1]):
    p = Rectangle((trans, -0.05), block_trans[i+1] - trans, 1.1, alpha=0.5,
                  color=block_colors[trial_blocks[trans]])
    ax2.add_patch(p)
# ax2.plot(np.convolve(decode_block['predictions'][0], np.ones(10), 'same') / 10, lw=2, color='k')
ax2.plot(np.convolve(decode_block['probabilities'][0], np.ones(10), 'same') / 10, lw=2, color='k')
# ax2.plot(decode_block['probabilities'][0], lw=2, color='k')

ax2.set(xlim=[0, trial_blocks.shape[0]], ylim=[-0.05, 1.05],
        ylabel='Decoding probability (10 trial rolling average)', xlabel='Trials')

plt.savefig(join(FIG_PATH, '%s_%s' % (REGION, DECODER)))

