# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

@author: guido
"""

from os import mkdir
from os.path import join, isdir
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
import seaborn as sns
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from ephys_functions import paths
from oneibl.one import ONE
one = ONE()

# Settings
BIN_SIZE = 50  # in ms
BIN_START = np.arange(-1050, -50, 50)  # ms relative to go cue
MIN_NEURONS = 20
SUBJECT = 'ZM_2240'
DATE = '2020-01-23'
PROBE = '00'
OVERWRITE = True


def exponential_decay(x, A, tau, B):
    y = (A * np.exp(-(x / tau))) + B
    return y


def _get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
    """
    Return the number of spikes in a sequence of time intervals, for each neuron.

    Parameters
    ----------
    spike_times : 1D array
        spike times (in seconds)
    spike_clusters : 1D array
        cluster ids corresponding to each event in `spikes`
    intervals : 2D array of shape (n_events, 2)
        the start and end times of the events

    Returns
    ---------
    counts : 2D array of shape (n_neurons, n_events)
        the spike counts of all neurons ffrom scipy.stats import sem, tor all events
        value (i, j) is the number of spikes of neuron `neurons[i]` in interval #j
    cluster_ids : 1D array
        list of cluster ids
    """

    # Check input
    assert intervals.ndim == 2
    assert intervals.shape[1] == 2
    assert np.all(np.diff(spike_times) >= 0), "Spike times need to be sorted"

    intervals_idx = np.searchsorted(spike_times, intervals)

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[intervals_idx[j, 0]:intervals_idx[j, 1]],
            minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids


# %%
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
        pop_vector1, cluster_ids = _get_spike_counts_in_bins(
                                            spikes[probe].times, spikes[probe].clusters, times1)

        # Get spike counts of all neurons during bin 2
        times2 = np.column_stack(((trials.goCue_times + bin2),
                                 (trials.goCue_times + (bin2 + BIN_SIZE_S))))
        pop_vector2, cluster_ids = _get_spike_counts_in_bins(
                                            spikes[probe].times, spikes[probe].clusters, times2)

        # Correlate the two bins for each neuron
        for n, cluster in enumerate(cluster_ids):

            # Correlate time bins
            corr_matrix[i, j, n], _ = pearsonr(pop_vector1[n], pop_vector2[n])

# Fill the diagonal with 0's instead of 1's
for n in range(corr_matrix.shape[2]):
    np.fill_diagonal(corr_matrix[:, :, n], 0)

# Get the brain regions
brain_region = []
for i, cluster in enumerate(cluster_ids):
    region = clusters[probe].acronym[clusters[probe].metrics.cluster_id == cluster][0]
    region = region.replace('/', '-')
    brain_region.append(region)

# Exclude regions with too few neurons
unique_regions, n_per_region = np.unique(brain_region, return_counts=True)
excl_regions = unique_regions[n_per_region < MIN_NEURONS]
incl_regions = [i for i, r in enumerate(brain_region) if r not in excl_regions]
corr_matrix = corr_matrix[:, :, incl_regions]
brain_region = np.array(brain_region)[incl_regions]
cluster_ids = cluster_ids[incl_regions]

# Average matrix over neurons in brain region and fit exponential decay at population level
timescale = dict()
auto_corr = pd.DataFrame()
for i, region in enumerate(np.unique(brain_region)):
    timescale[region] = dict()

    # Get average matrix
    mat = np.nanmean(corr_matrix[:, :, np.array(
                                [j for j, r in enumerate(brain_region) if r == region])], axis=2)
    timescale[region]['corr_matrix'] = mat

    # Get flattened vector from matrix
    corr_bin = []
    for j in range(1, mat.shape[0]):
        corr_bin.append(np.mean(np.diag(mat, j)))
    timescale[region]['decay_points'] = corr_bin

    # Fit exponential decay starting at the bin with maximum autocorrelation decay
    fit_start = np.argmin(np.diff(corr_bin))
    if fit_start > BIN_START.shape[0]/3:
        fit_start = np.argmax(corr_bin)
    if fit_start > BIN_START.shape[0]/3:
        fit_start = 0
    delta_time = np.arange(BIN_SIZE, BIN_SIZE*corr_matrix.shape[0], BIN_SIZE)
    fitted_params, _ = curve_fit(exponential_decay, delta_time[fit_start:],
                                 corr_bin[fit_start:], [0.5, 200, 0])
    timescale[region]['fit'] = fitted_params
    timescale[region]['time_constant'] = fitted_params[1]

# %% Plot

sns.set(style="ticks", context="paper", font_scale=1.5)

ax_grid = int(np.ceil(np.sqrt(len(timescale.keys()))))
f, ax = plt.subplots(ax_grid, ax_grid, sharex=True, sharey=True,
                     figsize=(ax_grid*3, ax_grid*3))
ax = np.reshape(ax, (1, ax_grid * ax_grid))[0]
for i, region in enumerate(timescale.keys()):
    sns.heatmap(timescale[region]['corr_matrix'], cbar=False, ax=ax[i])
    ax[i].set(xticks=np.arange(0, BIN_START.shape[0], 3), xticklabels=BIN_START[0:-1:3],
              yticks=np.arange(0, BIN_START.shape[0], 3),  yticklabels=BIN_START[0:-1:3],
              title=region)
    ax[i].tick_params(labelsize=12, labelrotation=45)
f.text(0.5, 0.02, 'Time to trial start (ms)', ha='center')
f.text(0.02, 0.5, 'Time to trial start (ms)', va='center', rotation='vertical')
plt.tight_layout(pad=1.8)
plt.savefig(join(FIG_PATH, '%s_%s_probe%s_corrmat' % (SUBJECT, DATE, PROBE)), dpi=300)

colors = sns.color_palette('colorblind', len(timescale.keys()))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

x = np.arange(0, BIN_START.shape[0] * BIN_SIZE, 1)
for i, region in enumerate(timescale.keys()):
    ax1.plot(x, exponential_decay(
                    x, timescale[region]['fit'][0], timescale[region]['fit'][1],
                    timescale[region]['fit'][2]), color=colors[i], label=region, lw=2)
    ax1.plot(delta_time, timescale[region]['decay_points'], 'o', color=colors[i])
ax1.legend(frameon=False)
ax1.set(ylabel='Auto-correlation', xlabel='\u0394 Time (ms)')

time_constant = []
for i, region in enumerate(timescale.keys()):
    time_constant.append(timescale[region]['time_constant'])

ax2.bar(np.arange(len(time_constant)), np.array(time_constant)[np.argsort(time_constant)],
        color=colors)
ax2.set(ylabel='Intrinsic timescale (ms)', xticks=np.arange(len(timescale.keys())),
        xticklabels=np.array(list(timescale.keys()))[np.argsort(time_constant)])
plt.xticks(rotation=45)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(FIG_PATH, '%s_%s_probe%s_timescale' % (SUBJECT, DATE, PROBE)), dpi=300)
