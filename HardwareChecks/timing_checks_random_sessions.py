#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:06:11 2019

@author: guido
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from scipy import stats
from os.path import join
import seaborn as sns
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
data_path = '/home/guido/Data/Behavior/'
fig_path = '/home/guido/Figures/Behavior/'
num_ses = 200

# Pick random sessions
sessions = subject.Subject * subject.SubjectLab * acquisition.Session & 'task_protocol LIKE "%training%"'
sessions_start = sessions.fetch('session_start_time')
sessions_select = np.random.choice(sessions_start, size=num_ses, replace=False)

timing_check = pd.DataFrame(columns=['subject_nickname','session_start_time','nan_stim_on','nan_go_cue','nan_response','neg_rt_go_cue','neg_rt_stim_on'])
for i in range(num_ses):
    print('Subject %d of %d'%(i+1,num_ses))
    # Load in trials
    trials_query = behavior.TrialSet.Trial * subject.Subject * subject.SubjectLab & 'session_start_time="%s"'%sessions_select[i]
    trials = pd.DataFrame(trials_query)
    
    try:
        timing_check.loc[i,'subject_nickname'] = trials.loc[0,'subject_nickname']
        timing_check.loc[i,'session_start_time'] = trials.loc[0,'session_start_time']
        timing_check.loc[i,'trial_number'] = len(trials)
        timing_check.loc[i,'nan_stim_on'] = sum(np.isnan(trials['trial_stim_on_time']))
        timing_check.loc[i,'nan_go_cue'] = sum(np.isnan(trials['trial_go_cue_time']))
        timing_check.loc[i,'nan_response'] = sum(np.isnan(trials['trial_response_time']))
        timing_check.loc[i,'neg_rt_go_cue'] = sum((trials['trial_response_time']-trials['trial_go_cue_time']) <= 0)
        timing_check.loc[i,'neg_rt_stim_on'] = sum((trials['trial_response_time']-trials['trial_stim_on_time']) <= 0)
        timing_check.loc[i,'stim_onset_diff>50ms'] = sum(np.abs(trials['trial_go_cue_time']-trials['trial_stim_on_time']) >= 0.05)
        
        print('Mouse: %s'%trials.loc[0,'subject_nickname'])
        print('Number of NaNs in go cue: %d'%sum(np.isnan(trials['trial_go_cue_time'])))
        print('Number of NaNs in stim on: %d'%sum(np.isnan(trials['trial_stim_on_time'])))
        print('Number of NaNs in response: %d'%sum(np.isnan(trials['trial_response_time'])))
        print('Number of negative reaction times from go cue: %d'%sum((trials['trial_response_time']-trials['trial_go_cue_time']) <= 0))
        print('Number of negative reaction times from stim on: %d'%sum((trials['trial_response_time']-trials['trial_stim_on_time']) <= 0))
        print(' ')
    except:
        print('Error')          
    
    timing_check.to_pickle(join(data_path, 'timing_check_random'))
    
timing_check = pd.read_pickle(join(data_path, 'timing_check_random'))



f = plt.figure()
ax = plt.gca()
ax.scatter(timing_check['session_start_time'], 
           ((timing_check['nan_stim_on']+timing_check['nan_go_cue']+timing_check['stim_onset_diff>50ms']) / 
           timing_check['trial_number'])*100)
ax.plot([datetime.date(2019,7,8), datetime.date(2019,7,8)], [-0.2, 100], 'g')
ax.set(ylabel='Failure rate (%)')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)


f, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(10,10), sharex=True)
ax1.scatter(timing_check['session_start_time'], timing_check['nan_stim_on']+timing_check['nan_go_cue']+timing_check['nan_response'])
ax1.plot([datetime.date(2019,7,8), datetime.date(2019,7,8)], [-0.2, 10], 'g')
ax1.set(ylim=[-0.2,10], ylabel='Number of NaNs in events')
#plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

ax2.scatter(timing_check['session_start_time'], timing_check['neg_rt_go_cue'])
ax2.plot([datetime.date(2019,7,8), datetime.date(2019,7,8)], [-0.2, 10], 'g')
ax2.set(ylim=[-0.2,10], ylabel='Number of negative RTs from go cue')
#plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

ax3.scatter(timing_check['session_start_time'], timing_check['neg_rt_stim_on'])
ax3.plot([datetime.date(2019,7,10), datetime.date(2019,7,10)], [-0.2, 10], 'g')
ax3.set(ylim=[-0.2,10], ylabel='Number of negative RTs from stim on')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)

ax4.scatter(timing_check['session_start_time'], timing_check['stim_onset_diff>50ms'])
ax4.plot([datetime.date(2019,7,10), datetime.date(2019,7,10)], [0, 120], 'g')
ax4.set(ylim=[-1,120], ylabel='Go cue and stim onset differ > 50 ms')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=40)

plt.tight_layout(pad = 3)
plt.savefig(join(fig_path, 'timing_checks.pdf'), dpi=300)
plt.savefig(join(fig_path, 'timing_checks.png'), dpi=300)
