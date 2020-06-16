#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate intrinsic time constants of all brain regions in the dataset from Steinmetz et al. (2019)
Data is downloaded through the Open Neurophysiology Environment (ONE), installation instructions:
https://github.com/int-brain-lab/ibllib/tree/onelight/oneibl#one-light

by Guido Meijer
June 8, 2020
"""

import os
from os.path import join, normpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from oneibl.onelight import ONE

# Paths (change this)
FIG_PATH = '/home/guido/Figures/Ephys/TimeConstant'
RESULT_PATH = '/home/guido/Data/Ephys'
DATA_PATH = '/media/guido/data/Steinmetz'

# Settings
BIN_SIZE = 50  # in ms
BIN_START = np.arange(-1050, -50, 50)  # ms relative to go cue
MIN_NEURONS = 10


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
# Set up ONE
one = ONE()
one.set_figshare_url('https://figshare.com/articles/steinmetz/9974357')
one.set_download_dir(join(DATA_PATH, '{subject}', '{date}'))

# Query all sessions with ephys and behavior
sessions = one.search(['spikes', 'trials'])

# Loop over recording sessions
timeconstant = pd.DataFrame()
for i, ses in enumerate(sessions):
    print('Calculating time constants for session %d of %d' % (i+1, len(sessions)))

    # Get subject and date
    ses_path = normpath(ses)
    ses_date = ses_path.split(os.sep)[-2]
    subject = ses_path.split(os.sep)[-3]

    # Download and load in data
    spikes = one.load_object(ses, 'spikes')
    clusters = one.load_object(ses, 'clusters')
    channels = one.load_object(ses, 'channels')
    trials = one.load_object(ses, 'trials')

    # Only use single units
    good_clusters = [i for i, x in enumerate(clusters._phy_annotation) if x[0] >= 2]
    spike_times = spikes.times[np.isin(spikes.clusters, good_clusters)]
    spike_clusters = spikes.clusters[np.isin(spikes.clusters, good_clusters)]
    cluster_channels = clusters.peakChannel[good_clusters]

    # Convert into seconds
    BIN_SIZE_S = BIN_SIZE / 1000
    BIN_START_S = BIN_START / 1000

    # Pre-allocate 3D matrix (bin1 x bin2 x neuron)
    corr_matrix = np.empty((BIN_START.shape[0], BIN_START.shape[0],
                            np.unique(spike_clusters).shape[0]))

    # Correlate every timebin with every other timebin
    for i, bin1 in enumerate(BIN_START_S):
        for j, bin2 in enumerate(BIN_START_S):
            # Get spike counts of all neurons during bin 1
            times1 = np.column_stack(((trials.goCue_times + bin1),
                                     (trials.goCue_times + (bin1 + BIN_SIZE_S))))
            pop_vector1, cluster_ids = _get_spike_counts_in_bins(
                                                        spike_times, spike_clusters, times1)

            # Get spike counts of all neurons during bin 2
            times2 = np.column_stack(((trials.goCue_times + bin2),
                                     (trials.goCue_times + (bin2 + BIN_SIZE_S))))
            pop_vector2, cluster_ids = _get_spike_counts_in_bins(
                                                        spike_times, spike_clusters, times2)

            # Correlate the two bins for each neuron
            for n, cluster in enumerate(cluster_ids):

                # Correlate time bins
                corr_matrix[i, j, n], _ = pearsonr(pop_vector1[n], pop_vector2[n])

    # Get the brain region for each cluster
    brain_region = channels.brainLocation.allen_ontology[np.squeeze(
                                                            cluster_channels)].values.astype(str)

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
                                    index=[0], data={'subject': subject,
                                                     'date': ses_date,
                                                     'region': region,
                                                     'time_constant': fitted_params[1]}))
        except:
            continue

    timeconstant.to_csv(join(RESULT_PATH, 'time_constant_regions_steinmetz.csv'))

# %% Plot
sns.set(context='paper', style='whitegrid', font_scale=2)
timeconstant = pd.read_csv(join(RESULT_PATH, 'time_constant_regions_steinmetz.csv'))

# Exclude outliers
timeconstant = timeconstant[timeconstant['time_constant'] < 1000]

n_regions = timeconstant.groupby('region').size()[
                                    timeconstant.groupby('region').size() > 3].reset_index()
timeconstant = timeconstant[timeconstant['region'].isin(n_regions['region'])]
timeconstant = timeconstant.sort_values('time_constant', ascending=False)

sort_regions = timeconstant.groupby('region').mean().sort_values(
                        'time_constant', ascending=False).reset_index()['region']

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(x='time_constant', y='region', data=timeconstant,
            order=sort_regions, ci=68, color=[0.6, 0.6, 0.6], ax=ax1)
ax1.set(xlabel='Intrinsic timeconstant (ms)', ylabel='', xlim=[0, 450])
sns.despine(trim=False)
plt.savefig(join(FIG_PATH, 'time_constant_steinmetz'))
