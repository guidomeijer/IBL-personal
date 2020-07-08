# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

from oneibl.one import ONE
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from os.path import expanduser, join, dirname
import pandas as pd


def paths():
    if expanduser('~') == '/home/guido':
        data_path = '/media/guido/data/Flatiron/'
    else:
        data_path = join(expanduser('~'), 'Downloads', 'FlatIron')
    fig_path = join(expanduser('~'), 'Figures', 'Ephys')
    save_path = join(expanduser('~'), 'Data', 'Ephys')
    return data_path, fig_path, save_path


def figure_style(font_scale=2, despine=True, trim=True):
    """
    Set style for plotting figures
    """
    sns.set(style="ticks", context="paper", font_scale=font_scale)
    if despine is True:
        sns.despine(trim=trim)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()


def check_trials(trials):
    
    if trials is None:
        return False        
    if trials.probabilityLeft is None:
        return False
    if len(trials.probabilityLeft[0].shape) > 0:
        return False
    if trials.probabilityLeft[0] != 0.5:
        return False
    if ((not hasattr(trials, 'stimOn_times'))  
            or (len(trials.feedback_times) != len(trials.feedbackType))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        return False
    return True


def sessions():
    ses = pd.read_csv(join(dirname(__file__), 'sessions.csv'), dtype='str')
    return ses


def download_data(nickname, date):
    one = ONE()
    eid = one.search(subject=nickname, date_range=[date, date])
    assert len(eid) == 1
    dtypes = ['_iblrig_taskSettings.raw',
              'spikes.times',
              'spikes.clusters',
              'clusters.channels',
              'clusters.metrics',
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


def dataset_types_to_download():
    dtypes = ['_iblrig_taskSettings.raw',
              'spikes.times',
              'spikes.clusters',
              'clusters.channels',
              'clusters.metrics',
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
    return dtypes
