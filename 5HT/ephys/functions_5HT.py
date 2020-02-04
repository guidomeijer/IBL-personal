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
from sklearn.model_selection import KFold
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
    plt.tight_layout()
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
    frontal_sessions = pd.DataFrame(data={'lab': ['mainenlab',
                                                  'mainenlab',
                                                  'mainenlab',
                                                  'danlab',
                                                  'mainenlab'],
                                          'subject': ['ZM_2240',
                                                      'ZM_2240',
                                                      'ZM_2240',
                                                      'DY_011',
                                                      'ZM_2241'],
                                          'date': ['2020-01-21',
                                                   '2020-01-22',
                                                   '2020-01-23',
                                                   '2020-01-30',
                                                   '2020-01-27'],
                                          'probe': ['00',
                                                    '00',
                                                    '00',
                                                    '00',
                                                    '00']})
    control_sessions = pd.DataFrame(data={'lab': ['mainenlab',
                                                  'mainenlab',
                                                  'mainenlab'],
                                          'subject': ['ZM_2240',
                                                      'ZM_2240',
                                                      'ZM_2241'],
                                          'date': ['2020-01-22',
                                                   '2020-01-24',
                                                   '2020-01-28'],
                                          'probe': ['01',
                                                    '00',
                                                    '00']})
    return frontal_sessions, control_sessions


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

    Returns
    -------
    f1 : float
        The F1-score of the classification
    cm : 2D matrix
        The normalized confusion matrix

    """
    assert resp.shape[0] == labels.shape[0]

    kf = KFold(n_splits=num_splits, shuffle=True)
    y_pred = np.array([])
    y_true = np.array([])
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
    f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    return f1, cm
