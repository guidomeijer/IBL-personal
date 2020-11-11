#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import decode, _get_spike_counts_in_bins
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
import alf
from ephys_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'bayes'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Response histograms')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'all'  # all or aligned
N_SESSIONS = 10
METRIC = 'accuracy'

# %%


eid = '259927fd-7563-4b03-bc5d-17b4d0fa7a55'
probe = 'probe00'
region = 'ACAd'

# Load in data
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
ses_path = one.path_from_eid(eid)
trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

# Get trial vectors
incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
trial_times = trials.stimOn_times[incl_trials]
probability_left = trials.probabilityLeft[incl_trials]
trial_blocks = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

# Get clusters in this brain region
region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]

# Select spikes and clusters
spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                             clusters_in_region)]

# Get matrix of all neuronal responses
times = np.column_stack(((trial_times - PRE_TIME), (trial_times + POST_TIME)))
pop_vector, cluster_ids = _get_spike_counts_in_bins(spks_region, clus_region, times)
pop_vector = pop_vector.T

# Plot histograms
for k in range(pop_vector.shape[1]):
    f, ax1 = plt.subplots(1, 1)
    ax1.hist(pop_vector[trial_blocks == 0, k], label='L', histtype='step', lw=3)
    ax1.hist(pop_vector[trial_blocks == 1, k], label='R', histtype='step', lw=3)
    ax1.set(xlabel='Spike count per trial', ylabel='Count', title='Region: %s, neuron %d' %
            (region, cluster_ids[k]))
    plt.legend()
    plt.savefig(join(FIG_PATH, '%s_neuron%d_%s' % (region, cluster_ids[k], eid)))
    plt.close(f)


