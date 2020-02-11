# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

from oneibl.one import ONE
from os.path import expanduser, join
from pathlib import Path


def paths():
    if expanduser('~') == '/home/guido':
        data_path = '/media/guido/data/Flatiron/'
    else:
        data_path = join(expanduser('~'), 'Downloads', 'FlatIron')
    fig_path = join(expanduser('~'), 'Figures', 'Ephys')
    save_path = join(expanduser('~'), 'Data', 'Ephys')
    return data_path, fig_path, save_path


def one_session_path(eid):
    one = ONE()
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


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
