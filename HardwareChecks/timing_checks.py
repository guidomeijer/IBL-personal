#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 12:06:11 2019

@author: guido
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from os.path import join
import seaborn as sns
import datajoint as dj
from ibl_pipeline import subject, acquisition, action, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = '/home/guido/Figures/Behavior/'

# Query list of subjects
nickname = 'IBL_11'
session_start_time = '2019-07-19 13:07:26'

# Load in trials
trials = behavior.TrialSet.Trial * subject.Subject & 'session_start_time="%s"'%session_start_time
trials = pd.DataFrame(trials)

# Print some sanity checks
print('Mouse: %s\nSession date: %s'%(nickname,session_start_time))
print('Number of NaNs in go cue: %d'%sum(np.isnan(trials['trial_go_cue_time'])))
print('Number of NaNs in stim on: %d'%sum(np.isnan(trials['trial_stim_on_time'])))
print('Number of NaNs in response: %d'%sum(np.isnan(trials['trial_response_time'])))
print('Number of negative reaction times from go cue: %d'%sum((trials['trial_response_time']-trials['trial_go_cue_time']) <= 0))
print('Number of negative reaction times from stim on: %d'%sum((trials['trial_response_time']-trials['trial_stim_on_time']) <= 0))

# Remove trials with nans
trials_nonan = trials.copy()
trials_nonan = trials_nonan.loc[np.isfinite(trials['trial_go_cue_time'])]
trials_nonan = trials_nonan.loc[np.isfinite(trials['trial_stim_on_time'])]
trials_nonan = trials_nonan.loc[np.isfinite(trials['trial_response_time'])]

# Plot timing histograms
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
sns.distplot(trials_nonan['trial_response_time']-trials_nonan['trial_go_cue_time'], kde=False, rug=True, ax=ax1)
ax1.set(xlabel='Response time from go cue (s)', ylabel='Count')

sns.distplot(trials_nonan['trial_response_time']-trials_nonan['trial_stim_on_time'], kde=False, rug=True, ax=ax2)
ax2.set(xlabel='Response time from stimulus onset (s)', ylabel='Count')

sns.distplot((trials_nonan['trial_stim_on_time']-trials_nonan['trial_go_cue_time'])*1000, kde=False, rug=True, ax=ax3)
ax3.set(xlabel='Stimulus onset - Go cue (ms)', ylabel='Count')

sns.distplot((trials_nonan['trial_go_cue_time']-trials_nonan['trial_go_cue_trigger_time'])*1000, kde=False, rug=True, ax=ax4)
ax4.set(xlabel='Delay go cue trigger to actual presentation (ms)', ylabel='Count')

plt.savefig(join(fig_path, 'timing_check_%s_%s.pdf'%(nickname,session_start_time)), dpi=300)
