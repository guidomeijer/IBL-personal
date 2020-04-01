#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:00:46 2020

@author: guido
"""

import pandas as pd
import numpy as np
from os.path import join, realpath, dirname
import datajoint as dj
from ibl_pipeline import subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
Q = 4
DAYS = np.arange(1, 11)
SAVE_PATH = join(dirname(realpath(__file__)), 'data')

# Query all subjects with project ibl_neuropixel_brainwide_01 and get trained date
subj_crit = subject.Subject.aggr(
        (acquisition.Session * behavior_analysis.SessionTrainingStatus)
         & 'training_status="trained_1a" OR training_status="trained_1b"',
         'subject_nickname', date_criterion='min(date(session_start_time))')

# Query the training day at which criterion is reached
subj_crit_day = ((dj.U('subject_uuid', 'day_of_crit')
                  & (behavior_analysis.BehavioralSummaryByDate * subj_crit
                     & 'session_date=date_criterion').proj(day_of_crit='training_day'))
                 * subject.Subject).proj('subject_nickname')

# Query reaction times
rt_query = behavior_analysis.ReactionTime.proj('reaction_time',
                                               session_date='DATE(session_start_time)')

for i, day in enumerate(DAYS):
    print('Building dataset for day %d of %d' % (day, DAYS[-1]))

    # Get dataframe with behavioral data
    behav = (subj_crit_day * behavior_analysis.BehavioralSummaryByDate * rt_query
             & 'training_day="%s"' % str(day)).fetch(format='frame')
    behav = behav.reset_index()
    behav['median_rt'] = [np.mean(i) for i in behav['reaction_time']]

    # Bin training time
    behav['learning_speed'] = pd.qcut(behav['day_of_crit'], q=Q, labels=np.arange(Q))

    # Drop nans in rt
    behav = behav[behav['median_rt'].notnull()]

    # Get behavioral metrics per session
    for j, ses in enumerate(behav['session_start_time']):
        trials = (behavior.TrialSet.Trial
                  & 'session_start_time = "%s"' % ses).fetch(format='frame')
        trials = trials.reset_index()
        trials['stim_side'] = np.array(trials['trial_stim_contrast_left'] == 0, dtype=int)

        # Get fast and slow median reaction times
        rt = trials['trial_response_time'] - trials['trial_stim_on_time']
        behav.loc[j, 'rt_fast'] = np.median(rt[rt < 1])
        behav.loc[j, 'rt_slow'] = np.median(rt[rt > 1])

        # Get quiescent period
        behav.loc[j, 'quiescent_period'] = np.median(trials['trial_stim_on_time']
                                                     - trials['trial_start_time'])

        # Get repeated errors
        rep_errors = 0
        for t in range(trials.shape[0]):
            if ((t != 0) and (trials.loc[t, 'trial_feedback_type'] == -1)
                    and (trials.loc[t-1, 'trial_feedback_type'] == -1)
                    and (trials.loc[t, 'stim_side'] == trials.loc[t-1, 'stim_side'])):
                rep_errors = rep_errors + 1
        behav.loc[j, 'rep_errors'] = rep_errors

        # Get bias
        trials = trials[trials['trial_response_choice'] != 'No Go']
        bias = np.abs(((np.sum(trials['trial_response_choice'] == 'CW')
                        / np.sum(trials['trial_response_choice'] == 'CCW'))
                       / (np.sum(trials['trial_stim_contrast_left'] != 0)
                          / np.sum(trials['trial_stim_contrast_right'] != 0))))
        behav.loc[j, 'bias'] = bias

    # Save dataframe
    behav.to_csv(join(SAVE_PATH, 'training_day_%d.csv' % day))
