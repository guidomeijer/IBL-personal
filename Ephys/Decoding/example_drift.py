#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:25:04 2020

@author: guido
"""



from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import decode, _get_spike_counts_in_bins
import seaborn as sns
import alf
import scipy as sp
from ephys_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()


neuron=5

EID = '15f742e1-1043-45c9-9504-f1e8a53c1744'
REGION = 'SNr'
PROBE = 'probe01'
PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'bayes'
ITERATIONS = 1000
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Single sessions', DECODER)


def _generate_pseudo_blocks(n_trials, factor=60, min_=20, max_=100):
    block_ids = []
    while len(block_ids) < n_trials:
        x = np.random.exponential(factor)
        while (x <= min_) | (x >= max_):
            x = np.random.exponential(factor)
        if (len(block_ids) == 0) & (np.random.randint(2) == 0):
            block_ids += [0] * int(x)
        elif (len(block_ids) == 0):
            block_ids += [1] * int(x)
        elif block_ids[-1] == 0:
            block_ids += [1] * int(x)
        elif block_ids[-1] == 1:
            block_ids += [0] * int(x)
    return np.array(block_ids[:n_trials])


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

# Get matrix of all neuronal responses
times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
pop_vector, cluster_ids = _get_spike_counts_in_bins(spks_region, clus_region, times)
pop_vector = pop_vector.T

# Phase randomize
if pop_vector.shape[0] % 2 == 0:
        pop_vector = pop_vector[:-1]
rand_pop_vector = np.empty(pop_vector.shape)
frequencies = int((pop_vector.shape[0] - 1) / 2)
fsignal = sp.fft.fft(pop_vector, axis=0)
power = np.abs(fsignal[1:1 + frequencies])
phases = 2 * np.pi * np.random.rand(frequencies)
for k in range(pop_vector.shape[1]):
    newfsignal = fsignal[0, k]
    newfsignal = np.append(newfsignal, np.exp(1j * phases) * power[:, k])
    newfsignal = np.append(newfsignal, np.flip(np.exp(-1j * phases) * power[:, k]))
    newsignal = sp.fft.ifft(newfsignal)
    rand_pop_vector[:, k] = np.abs(newsignal.real)

# %%
figure_style(font_scale=1.8)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=False, dpi=150)
ax1.plot(pop_vector[:,neuron], lw=2)
ax1.set(xlabel='Trials', ylabel='Spike count', xlim=[0, 800], ylim=[0, 30])

ax2.plot(pop_vector[:,neuron], lw=2)
train = np.random.choice(pop_vector.shape[0], size=int(pop_vector.shape[0] * 0.08), replace=False)
ax2.scatter(train, np.ones(train.shape) * np.max(pop_vector[:,neuron]) + 2, color='red', s=8)
ax2.set(xlabel='Trials', ylabel='Spike count', xlim=[0, 800], ylim=[0, 30])

ax3.plot(pop_vector[:,neuron], lw=2)
ax3.plot(rand_pop_vector[:,neuron], lw=3)
ax3.set(xlabel='Trials', ylabel='Spike count', xlim=[0, 800], ylim=[0, 30])

plt.tight_layout()
sns.despine()

#%%
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=150)
ax1.plot(range(100), np.log10(power[:100, neuron]), lw=2)
ax1.set(ylabel='log10 Power', xlabel='Frequencies (Hz)')
plt.tight_layout()
sns.despine()

# %%


figure_style(font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(5, 5), sharey=True, dpi=150)
ax1.plot(trials.probabilityLeft, lw=2)
ax1.set(ylabel='Probability left stimulus', xlabel='Trials', title='Actual blocks')


figure_style(font_scale=1.8)
f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 5), sharey=True,
                                                     sharex=True, dpi=150)
fake_blocks = _generate_pseudo_blocks(trials.probabilityLeft.shape[0]-90)
fake_blocks = fake_blocks + 0.2
fake_blocks[fake_blocks == 1.2] = 0.8
fake_blocks = np.concatenate((np.ones(90)*0.5, fake_blocks))
ax1.plot(fake_blocks, lw=2)
ax1.set(ylabel='P(left)', xlabel='Trials')


fake_blocks = _generate_pseudo_blocks(trials.probabilityLeft.shape[0]-90)
fake_blocks = fake_blocks + 0.2
fake_blocks[fake_blocks == 1.2] = 0.8
fake_blocks = np.concatenate((np.ones(90)*0.5, fake_blocks))
ax2.plot(fake_blocks, lw=2)
ax2.set(ylabel='P(left)', xlabel='Trials')


fake_blocks = _generate_pseudo_blocks(trials.probabilityLeft.shape[0]-90)
fake_blocks = fake_blocks + 0.2
fake_blocks[fake_blocks == 1.2] = 0.8
fake_blocks = np.concatenate((np.ones(90)*0.5, fake_blocks))
ax3.plot(fake_blocks, lw=2)
ax3.set(ylabel='P(left)', xlabel='Trials')


fake_blocks = _generate_pseudo_blocks(trials.probabilityLeft.shape[0]-90)
fake_blocks = fake_blocks + 0.2
fake_blocks[fake_blocks == 1.2] = 0.8
fake_blocks = np.concatenate((np.ones(90)*0.5, fake_blocks))
ax4.plot(fake_blocks, lw=2)
ax4.set(ylabel='P(left)', xlabel='Trials')


fake_blocks = _generate_pseudo_blocks(trials.probabilityLeft.shape[0]-90)
fake_blocks = fake_blocks + 0.2
fake_blocks[fake_blocks == 1.2] = 0.8
fake_blocks = np.concatenate((np.ones(90)*0.5, fake_blocks))
ax5.plot(fake_blocks, lw=2)
ax5.set(ylabel='P(left)', xlabel='Trials')


fake_blocks = _generate_pseudo_blocks(trials.probabilityLeft.shape[0]-90)
fake_blocks = fake_blocks + 0.2
fake_blocks[fake_blocks == 1.2] = 0.8
fake_blocks = np.concatenate((np.ones(90)*0.5, fake_blocks))
ax6.plot(fake_blocks, lw=2)
ax6.set(ylabel='P(left)', xlabel='Trials')

