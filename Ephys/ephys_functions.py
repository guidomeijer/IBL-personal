# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:22:01 2020

@author: guido
"""

from oneibl.one import ONE
import pandas as pd


def data_path():
    path = 'C:\\Users\\guido\\Downloads\\FlatIron\\mainenlab\\Subjects'
    return path


def download_data(nickname, date):
    one = ONE()
    eid = one.search(subject=nickname, date_range=[date, date])
    assert len(eid) == 1
    dtypes = ['_iblrig_taskSettings.raw',
              'spikes.times',
              'spikes.clusters',
              'clusters.probes',
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
              'trials.reponse_times',
              'trials.stimOn_times']
    one.load(eid[0], dataset_types=dtypes, download_only=True)
    # one.load(eid[0], clobber=False, download_only=True)


def frontal_sessions():
    sessions = pd.DataFrame(data={'subject': ['ZM_1897'],
                                  'date': ['2019-12-06'],
                                  'probe': ['00']})
    return sessions

