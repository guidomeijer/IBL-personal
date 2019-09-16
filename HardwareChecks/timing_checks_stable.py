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
from os.path import join
import seaborn as sns
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
data_path = '/home/guido/Data/Behavior/'
fig_path = '/home/guido/Figures/Behavior/'

# Pick random sessions
sessions = subject.Subject * subject.SubjectLab * acquisition.Session \
                    & 'task_protocol LIKE "%training%"' & 'DATE(session_start_time) > "2019-07-10"'
sessions_start = sessions.fetch('session_start_time')

timing_check = pd.DataFrame(columns=['subject_nickname','session_start_time','num_trials','nan_stim_on','nan_go_cue',\
                                             'nan_response','neg_rt_go_cue','neg_rt_stim_on','audio>50ms'])
for i, session_start in enumerate(sessions_start):
    print('Subject %d of %d'%(i+1,len(sessions_start)))
    # Load in trials
    trials_query = behavior.TrialSet.Trial * subject.Subject * subject.SubjectLab & 'session_start_time="%s"'%session_start
    trials = pd.DataFrame(trials_query)
    
    try:
        timing_check.loc[i,'subject_nickname'] = trials.loc[0,'subject_nickname']
        timing_check.loc[i,'session_start_time'] = trials.loc[0,'session_start_time']
        timing_check.loc[i,'num_trials'] = len(trials)
        timing_check.loc[i,'nan_stim_on'] = sum(np.isnan(trials['trial_stim_on_time']))
        timing_check.loc[i,'nan_go_cue'] = sum(np.isnan(trials['trial_go_cue_time']))
        timing_check.loc[i,'nan_response'] = sum(np.isnan(trials['trial_response_time']))
        timing_check.loc[i,'neg_rt_go_cue'] = sum((trials['trial_response_time']-trials['trial_go_cue_time']) <= 0)
        timing_check.loc[i,'neg_rt_stim_on'] = sum((trials['trial_response_time']-trials['trial_stim_on_time']) <= 0)
        timing_check.loc[i,'stim_onset_diff>50ms'] = sum(np.abs(trials['trial_go_cue_time']-trials['trial_stim_on_time']) >= 0.05)
        timing_check.loc[i,'audio>100ms'] = sum(trials['trial_go_cue_time']-trials['trial_go_cue_trigger_time'] >= 0.1)
        
        print('Mouse: %s'%trials.loc[0,'subject_nickname'])
        print('Number of NaNs in go cue: %d'%sum(np.isnan(trials['trial_go_cue_time'])))
        print('Number of NaNs in stim on: %d'%sum(np.isnan(trials['trial_stim_on_time'])))
        print('Number of NaNs in response: %d'%sum(np.isnan(trials['trial_response_time'])))
        print('Number of negative reaction times from go cue: %d'%sum((trials['trial_response_time']-trials['trial_go_cue_time']) <= 0))
        print('Number of negative reaction times from stim on: %d'%sum((trials['trial_response_time']-trials['trial_stim_on_time']) <= 0))
        print('Number of go cue and stim onset difference > 50 ms: %d'%sum(np.abs(trials['trial_go_cue_time']-trials['trial_stim_on_time']) >= 0.05))
        print('Number of audio > 100 ms: %d'%sum(trials['trial_go_cue_time']-trials['trial_go_cue_trigger_time'] >= 0.1))
        print(' ')
    except:
        print('Error')          
    
    #timing_check.to_pickle(join(data_path, 'timing_check_stable'))
    
timing_check = pd.read_pickle(join(data_path, 'timing_check_stable'))
failure_rate = ((timing_check['nan_stim_on']+timing_check['nan_go_cue']+timing_check['stim_onset_diff>50ms']) / 
           timing_check['num_trials'])*100

f = plt.figure()
ax = plt.gca()
ax.scatter(timing_check['session_start_time'], failure_rate)
ax.set(ylabel='Failure rate (%)', xlim=[datetime.date(2019,7,10), datetime.date(2019,8,1)])
ax.plot([datetime.date(2019,7,10), datetime.date(2019,8,1)],[5,5], 'r')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=40)
plt.tight_layout(pad = 3)


f, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2, 2, figsize=(10,10), sharex=True)
ax1.scatter(timing_check['session_start_time'], timing_check['nan_stim_on']+timing_check['nan_go_cue']+timing_check['nan_response'])
ax1.plot([datetime.date(2019,7,10), datetime.date(2019,7,10)], [-0.2, 10], 'g')
ax1.set(ylim=[-0.2,10], ylabel='Number of NaNs in events')
#plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

ax2.scatter(timing_check['session_start_time'], timing_check['neg_rt_go_cue'])
ax2.plot([datetime.date(2019,7,10), datetime.date(2019,7,10)], [-0.2, 10], 'g')
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
plt.savefig(join(fig_path, 'timing_checks_stable.pdf'), dpi=300)
plt.savefig(join(fig_path, 'timing_checks_stable.png'), dpi=300)
