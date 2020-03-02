# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

from oneibl.one import ONE
from os.path import expanduser, join
import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix


def paths():
    if expanduser('~') == '/home/guido':
        data_path = '/media/guido/data/Flatiron/'
    else:
        data_path = join(expanduser('~'), 'Downloads', 'FlatIron')
    fig_path = join(expanduser('~'), 'Figures', '5HT', 'ephys')
    save_path = join(expanduser('~'), 'Data', '5HT')
    return data_path, fig_path, save_path


def plot_settings():
    sns.set(style="ticks", context="paper", font_scale=1.4)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


def download_data(nickname, date):
    one = ONE()
    eid = one.search(subject=nickname, date_range=[date, date])
    assert len(eid) == 1
    dtypes = ['_iblrig_taskSettings.raw',
              'spikes.times',
              'spikes.clusters',
              'clusters.channels',
              'clusters.metrics',
              'clusters.depths',
              'clusters.probes',
              'probes.trajectory',
              'trials.choice',
              'trials.intervals',
              'trials.contrastLeft',
              'trials.contrastRight',
              'trials.feedback_times',
              'trials.goCue_times',
              'trials.feedbackType',
              'trials.probabilityLeft',
              'trials.response_times',
              'trials.stimOn_times']
    one.load(eid[0], dataset_types=dtypes, download_only=True)


def sessions():
    ses_path = join(pathlib.Path(__file__).parent.absolute(), 'sessions')
    frontal_sessions = pd.read_csv(join(ses_path, 'frontal_sessions.csv'), dtype=str)
    control_sessions = pd.read_csv(join(ses_path, 'control_sessions.csv'), dtype=str)
    return frontal_sessions, control_sessions


def one_session_path(eid):
    one = ONE()
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


def decoding(resp, labels, clf, num_splits):
    """
    Parameters
    ----------
    resp : TxN matrix
        Neuronal responses of N neurons in T trials
    labels : 1D array
        Class labels for T trials
    clf : object
        sklearn decoder object
    NUM_SPLITS : int
        The n in n-fold cross validation
        input 1 for leave-one-out cross-validation

    Returns
    -------
    f1 : float
        The F1-score of the classification
    cm : 2D matrix
        The normalized confusion matrix

    """
    assert resp.shape[0] == labels.shape[0]

    # Initialize cross-validation
    if num_splits == 1:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=num_splits, shuffle=True)

    # Loop over splits into training and testing
    y_pred = np.zeros(labels.shape)
    y_probs = np.zeros(labels.shape)
    for train_index, test_index in cv.split(resp):

        # Fit the model to the training data
        clf.fit(resp[train_index], [labels[j] for j in train_index])

        # Predict the test data
        y_pred[test_index] = clf.predict(resp[test_index])

        # Get the probability of the prediction for ROC analysis
        probs = clf.predict_proba(resp[test_index])
        y_probs[test_index] = probs[:, 1]  # keep positive only

    # Calculate performance metrics and confusion matrix
    f1 = f1_score(labels, y_pred)
    auroc = roc_auc_score(labels, y_probs)
    cm = confusion_matrix(labels, y_pred)

    return f1, auroc, cm


def get_spike_counts_in_bins(spike_times, spike_clusters, intervals):
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

    # For each neuron and each interval, the number of spikes in the interval.
    cluster_ids = np.unique(spike_clusters)
    n_neurons = len(cluster_ids)
    n_intervals = intervals.shape[0]
    counts = np.zeros((n_neurons, n_intervals), dtype=np.uint32)
    for j in range(n_intervals):
        t0, t1 = intervals[j, :]
        # Count the number of spikes in the window, for each neuron.
        x = np.bincount(
            spike_clusters[(t0 <= spike_times) & (spike_times < t1)],
            minlength=cluster_ids.max() + 1)
        counts[:, j] = x[cluster_ids]
    return counts, cluster_ids
