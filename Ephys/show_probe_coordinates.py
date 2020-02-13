#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:34:22 2019

@author: guido
"""

from oneibl.one import ONE
one = ONE()

mouse_id = 'ZM_2407'

dtypes = ['_iblrig_taskSettings.raw', 'probes.trajectory', 'trials.feedback_times']
eids, ses_info = one.search(subject=mouse_id, dataset_types=dtypes, details=True)

print('\nMouse ID: %s' % mouse_id)
for i, eid in enumerate(eids):
    d = one.load(eid, dataset_types=dtypes, download_only=False, dclass_output=False)
    if len(d) == 3:
        trials = d[2]
        traj = d[1]
    elif len(d) == 4:
        traj = d[2]
        trials = d[3]
    if trials is None:
        continue

    print('\nSession date: %s\nSession eid: %s' % (ses_info[i]['start_time'][:10], eid))
    print('%d trials' % len(trials))
    for p in range(len(traj)):
        print('\n%s\nAP: %s\nML: %s\nDepth: %s\nPhi: %s\nTheta: %s' % (traj[p]['label'],
                                                   round(traj[p]['y']/1000, 2),
                                                   round(traj[p]['x']/1000, 2),
                                                   round(traj[p]['depth']/1000, 1),
                                                   round(traj[p]['phi'], 1),
                                                   round(traj[p]['theta'], 1)))
