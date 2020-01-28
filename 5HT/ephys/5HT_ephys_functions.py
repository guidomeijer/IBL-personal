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


def frontal_sessions():
    sessions = pd.DataFrame(data={'subject': ['ZM_1897'],
                                  'date': ['2019-12-06'],
                                  'probe': ['00']})
    return sessions



