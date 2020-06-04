# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
from brainbox.population import _get_spike_counts_in_bins as spike_bins
import brainbox.io.one as bbone
import seaborn as sns
import shutil
import numpy as np
from scipy.stats import pearsonr
from ephys_functions import paths
from brainbox.io.one import load_spike_sorting, load_channel_locations
from oneibl.one import ONE
one = ONE()

# Settings
BIN_SIZE = 50  # in ms
BIN_START = np.arange(-550, -50, 50)  # ms relative to go cue
SUBJECT = 'ZM_2240'
DATE = '2020-01-23'
PROBE = '00'
OVERWRITE = True

# Set path to save plots
DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'TimeConstant')

# Load in data
eids = one.search(subject=SUBJECT, date_range=DATE)
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eids[0], one=one)
trials = one.load_object(eids[0], 'trials')

# Only use single units
probe = 'probe' + PROBE
spikes[probe].times = spikes[probe].times[np.isin(
        spikes[probe].clusters, clusters[probe].metrics.cluster_id[
            clusters[probe].metrics.ks2_label == 'good'])]
spikes[probe].clusters = spikes[probe].clusters[np.isin(
        spikes[probe].clusters, clusters[probe].metrics.cluster_id[
            clusters[probe].metrics.ks2_label == 'good'])]

# Make directory
if (isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
        and (OVERWRITE is True)):
    shutil.rmtree(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))
if not isdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe))):
    mkdir(join(FIG_PATH, '%s_%s_%s' % (SUBJECT, DATE, probe)))

# Loop over neurons
for n, cluster in enumerate(np.unique(spikes[probe].clusters)):

    # Get brain region of neuron
    region = channels[probe].acronym[clusters[probe].channels[
                         clusters[probe].metrics.cluster_id == cluster][0]]
    region = region.replace('/', '-')

# Convert into seconds
BIN_SIZE_S = BIN_SIZE / 1000
BIN_START_S = BIN_START / 1000

# Pre-allocate 3D matrix (bin1 x bin2 x neuron)
corr_matrix = np.empty((BIN_START.shape[0], BIN_START.shape[0],
                        np.unique(spikes[probe].clusters).shape[0]))

# Correlate every timebin with every other timebin
for i, bin1 in enumerate(BIN_START_S):
    print('Correlating bin %d of %d with every other bin' % (i+1, BIN_START_S.shape[0]))
    for j, bin2 in enumerate(BIN_START_S):
        # Get spike counts of all neurons during bin 1
        times1 = np.column_stack(((trials.goCue_times + bin1),
                                 (trials.goCue_times + (bin1 + BIN_SIZE_S))))
        pop_vector1, cluster_ids = spike_bins(spikes[probe].times, spikes[probe].clusters, times1)

        # Get spike counts of all neurons during bin 2
        times2 = np.column_stack(((trials.goCue_times + bin2),
                                 (trials.goCue_times + (bin2 + BIN_SIZE_S))))
        pop_vector2, cluster_ids = spike_bins(spikes[probe].times, spikes[probe].clusters, times2)

        # Correlate the two bins for each neuron
        for n, cluster in enumerate(cluster_ids):

            # Correlate time bins
            corr_matrix[i, j, n], _ = pearsonr(pop_vector1[n], pop_vector2[n])

# Get the brain regions
brain_region = []
for n, cluster in enumerate(cluster_ids):
    region = clusters[probe].acronym[clusters[probe].metrics.cluster_id == cluster][0]
    region = region.replace('/', '-')
    brain_region.append(region)

