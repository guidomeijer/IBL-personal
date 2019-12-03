# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:55:33 2019

List of plotting functions for DLC data

@author: Guido, Kelly
"""

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore


def peri_plot(trace, timestamps, events, ax, time_win, norm='none'):
    """
    Plot a peri-plot centered around a behavioral event
    :trace:        1D array with trace to be plotted
    :timestamps:   1D array with timestamps in seconds
    :events:       1D array with event times to center around
    :ax:           Axes to plot in, set to None to skip plotting
    :time_window:  time window in seconds
    :norm:         how to perform normalization
                   'none': no normalization (default)
                   'zscore': z-score the entire trace
                   'baseline': subtract the baseline from each trial trace
                               defined as the time before onset of the event
    """

    # Check input
    if np.size(time_win) != 2 or time_win[0] > 0 or time_win[1] < 0:
        raise Exception('time_win should be two element vector of which the first element ' \
                        'is negative and the second element positive')
    if norm != 'none' and norm != 'zscore' and norm != 'baseline':
        raise Exception('norm should be none, zscore or baseline')

    # Transform time window into samples
    sampling_rate = 1 / np.mean(np.diff(timestamps))
    sample_win = [np.int(np.round(time_win[0] * sampling_rate)),
                  np.int(np.round(time_win[1] * sampling_rate))]
    time_trace = np.linspace(time_win[0], time_win[1], np.sum(np.abs(sample_win)))

    # Z-score entire trace
    if norm == 'zscore':
        trace = zscore(trace)

    # Create dataframe for line plot
    peri_df = pd.DataFrame(columns=['event_nr', 'timepoint', 'trace'])
    for i in np.arange(np.size(events)):
        if (np.argmin(np.abs(timestamps-events[i]))+sample_win[0] > 0 and
                np.argmin(np.abs(timestamps-events[i]))+sample_win[1] < np.size(trace)):

            # Get trace for this trial
            this_trace = trace[np.argmin(np.abs(timestamps-events[i])) + sample_win[0]:
                               np.argmin(np.abs(timestamps-events[i])) + sample_win[1]]

            # Perform baseline correction
            if norm == 'baseline':
                this_trace = this_trace - np.median(this_trace[time_trace < time_win[0]/2])

            # Add to dataframe
            this_df = pd.DataFrame(data={'event_nr': np.ones(np.size(this_trace),
                                                             dtype=int)*(i+1),
                                         'timepoint': time_trace, 'trace': this_trace})
            peri_df = pd.concat([peri_df, this_df], ignore_index=True)
    return peri_df

    # Plot
    if ax is not None:
        sns.lineplot(x='timepoint', y='trace', data=peri_df, ci=68, ax=ax)
