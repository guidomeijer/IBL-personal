# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

from oneibl.one import ONE
from os.path import expanduser, join
import pandas as pd


def paths():
    if expanduser('~') == '/home/guido':
        data_path = '/media/guido/data/Flatiron/mainenlab/Subjects'
    else:
        data_path = join(expanduser('~'), 'Downloads', 'FlatIron', 'mainenlab', 'Subjects')
    fig_path = join(expanduser('~'), 'Figures', '5HT', 'ephys')
    return data_path, fig_path


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


def sessions():
    frontal_sessions = pd.DataFrame(data={'subject': ['ZM_2240', 'ZM_2240', 'ZM_2240'],
                                          'date': ['2020-01-21', '2020-01-22', '2020-01-23'],
                                          'probe': ['00', '00', '00']})
    control_sessions = pd.DataFrame(data={'subject': ['ZM_2240', 'ZM_2240'],
                                          'date': ['2020-01-22', '2020-01-24'],
                                          'probe': ['01', '00']})
    return frontal_sessions, control_sessions
