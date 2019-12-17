#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:28:14 2019

@author: guido
"""

import numpy as np
import pandas as pd
import datajoint as dj
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import subject, acquisition, behavior
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
TRAINING_DAYS = [3, 5]

# Query all subjects with project ibl_neuropixel_brainwide_01 and get trained date
subj_crit = subject.Subject.aggr(
        (acquisition.Session * behavior_analysis.SessionTrainingStatus)
         & 'training_status="trained_1a" OR training_status="trained_1b"',
         'subject_nickname', date_criterion='min(date(session_start_time))')

# Query the training day at which criterion is reached
subj_crit_day = (dj.U('subject_uuid', 'day_of_crit')
                 & (behavior_analysis.BehavioralSummaryByDate * subj_crit
                    & 'session_date=date_criterion').proj(day_of_crit='training_day'))

# Query days around the day at which criterion is reached
days = (behavior_analysis.BehavioralSummaryByDate
        * subject.Subject
        * subj_crit.proj('subject_uuid')
        & ('training_day between %d and %d' % (TRAINING_DAYS[0], TRAINING_DAYS[1]))).proj(
               'subject_uuid', 'subject_nickname', 'session_date')

# Use dates to query sessions
ses_query = (acquisition.Session).aggr(
        days, from_date='min(session_date)', to_date='max(session_date)')
sessions = (acquisition.Session * ses_query & 'date(session_start_time) >= from_date'
            & 'date(session_start_time) <= to_date')

# Loop over subjects
for i, nickname in enumerate(np.unique(sessions.fetch('subject_nickname'))):
    if np.mod(i+1, 10) == 0:
        print('Loading data of subject %d of %d' % (i+1, len(
                np.unique(sessions.fetch('subject_nickname')))))

    # Get trials of
    trials = (sessions * behavior.TrialSet.Trial
              & 'subject_nickname = "%s"' % nickname).fetch(format='frame')
    trials = trials.reset_index()

    # Fit a psychometric function to these trials and get fit results
    fit_df = dj2pandas(trials)
    fit_result = fit_psychfunc(fit_df)

    # Get RT, performance and number of trials
    reaction_time = trials['rt'].median()*1000
    perf_easy = trials['correct_easy'].mean()*100
    ntrials_perday = trials.groupby('session_uuid').count()['trial_id'].mean()