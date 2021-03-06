#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Decode left/right block identity from all brain regions

@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from ephys_functions import paths, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
BIN_SIZE = 50  # in ms
BIN_START = np.arange(-1050, -50, 50)  # ms relative to go cue
MIN_NEURONS = 10
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')


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
# Get list of all recordings that have histology
rec_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
recordings = pd.DataFrame(data={
                            'eid': [rec['session']['id'] for rec in rec_with_hist],
                            'probe': [rec['probe_name'] for rec in rec_with_hist],
                            'date': [rec['session']['start_time'][:10] for rec in rec_with_hist],
                            'subject': [rec['session']['subject'] for rec in rec_with_hist]})

# Get list of eids of ephysChoiceWorld sessions
eids = one.search(dataset_types=['spikes.times', 'probes.trajectory'],
                  task_protocol='_iblrig_tasks_ephysChoiceWorld')

# Select only the ephysChoiceWorld sessions and sort by eid
recordings = recordings[recordings['eid'].isin(eids)]
recordings = recordings.sort_values('eid').reset_index()

timeconstant = pd.DataFrame()
for i, eid in enumerate(recordings['eid'].values):

    # Load in data (only when not already loaded from other probe)
    print('Processing recording %d of %d' % (i+1, len(recordings)))
    if i == 0:
        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
            trials = one.load_object(eid, 'trials')
        except:
            continue
    elif recordings.loc[i-1, 'eid'] != recordings.loc[i, 'eid']:
        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
            trials = one.load_object(eid, 'trials')
        except:
            continue

    # Get probe
    probe = recordings.loc[i, 'probe']
    if probe not in spikes.keys():
        continue

    # Only use single units
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
    for i, region in enumerate(np.unique(brain_region)):
        # Get average matrix
        mat = np.nanmean(corr_matrix[:, :, np.array(
                            [j for j, r in enumerate(brain_region) if r == region])], axis=2)

        # Get flattened vector from matrix
        corr_bin = []
        for j in range(1, mat.shape[0]):
            corr_bin.append(np.mean(np.diag(mat, j)))

        # Fit exponential decay starting at the bin with maximum autocorrelation decay, if that
        # doesn't work start at the max autocorrelation, if that doesn't work start at beginning
        fit_start = np.argmin(np.diff(corr_bin))
        if fit_start > BIN_START.shape[0]/3:
            fit_start = np.argmax(corr_bin)
        if fit_start > BIN_START.shape[0]/3:
            fit_start = 0
        delta_time = np.arange(BIN_SIZE, BIN_SIZE*corr_matrix.shape[0], BIN_SIZE)
        try:
            fitted_params, _ = curve_fit(exponential_decay, delta_time[fit_start:],
                                         corr_bin[fit_start:], [0.5, 200, 0])
            timeconstant = timeconstant.append(pd.DataFrame(
                                    index=[0], data={'subject': recordings.loc[i, 'subject'],
                                                     'date': recordings.loc[i, 'date'],
                                                     'eid': recordings.loc[i, 'eid'],
                                                     'probe': probe,
                                                     'region': region,
                                                     'time_constant': fitted_params[1]}))
        except:
            continue

    timeconstant.to_csv(join(SAVE_PATH, 'time_constant_regions.csv'))

# %% Plot
timeconstant = pd.read_csv(join(SAVE_PATH, 'time_constant_regions.csv'))

n_regions = timeconstant.groupby('region').size()[
                                    timeconstant.groupby('region').size() > 3].reset_index()
timeconstant = timeconstant[timeconstant['region'].isin(n_regions['region'])]
timeconstant = timeconstant.sort_values('time_constant', ascending=False)

sort_regions = timeconstant.groupby('region').mean().sort_values(
                        'time_constant', ascending=False).reset_index()['region']

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))

sns.barplot(x='time_constant', y='region', data=timeconstant, order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Intrinsic timeconstant (ms)', ylabel='', xscale='log')
figure_style(font_scale=1.2)
