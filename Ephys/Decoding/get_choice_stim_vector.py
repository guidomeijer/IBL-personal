#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 13:44:16 2021

@author: guido
"""

import pandas as pd
from os.path import join
from my_functions import paths, query_sessions, check_trials, load_trials
from oneibl.one import ONE
one = ONE()

# Settings
INCL_SESSIONS = 'aligned-behavior'

# Query sessions
eids, _ = query_sessions(selection=INCL_SESSIONS)

# Loop over sessions
all_trials = pd.DataFrame()
for i, eid in enumerate(eids):
    print(f'Loading trails of session {i+1} of {len(eids)}')
    try:
        trials = load_trials(eid)
    except:
        continue
    if check_trials(trials):
        ses_info = one.get_details(eid)
        trials['subject'] = ses_info['subject']
        trials['date'] = ses_info['start_time'][:10]
        trials = trials.drop(columns=['stimOn_times', 'feedback_times',
                                      'goCue_times', 'right_choice'])
        all_trials = all_trials.append(trials[trials['probabilityLeft'] != 0.5])
        print(f'Added {len(trials)} trial (total {len(all_trials)})')
print('Saving results..')
all_trials.to_pickle(join(paths()[2], 'Ephys', 'Decoding', 'all_trials.p'))
print('Done')