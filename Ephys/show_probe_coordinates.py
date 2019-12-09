#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:34:22 2019

@author: guido
"""

from oneibl.one import ONE
one = ONE()

mouse_id = 'ZM_1897'

eids, ses_info = one.search(subject=mouse_id, dataset_types='spikes.times', details=True)
dtypes = ['_iblrig_taskSettings.raw', 'probes.trajectory', 'trials.feedback_times']

print('\nMouse ID: %s' % mouse_id)
for i, eid in enumerate(eids):
    d, traj, trials = one.load(eid, dataset_types=dtypes, download_only=False, dclass_output=False)
    if d is None:
        continue

    print('\nSession date: %s\nSession eid: %s' % (ses_info[i]['start_time'][:10], eid))
    print('%d trials' % len(trials))
    for p in range(len(traj)):
        print('\n%s\nAP: %s\nML: %s\nDepth: %s' % (traj[p]['label'],
                                                   round(traj[p]['y']/1000, 1),
                                                   round(traj[p]['x']/1000, 1),
                                                   round(traj[p]['depth']/1000, 1)))
