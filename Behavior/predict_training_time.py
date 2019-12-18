#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:28:14 2019

@author: guido
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from dj_tools import dj2pandas, fit_psychfunc
from ibl_pipeline import subject, acquisition, behavior, reference
from ibl_pipeline.analyses import behavior as behavior_analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix


# Decoding function with n-fold cross validation
def decoding(resp, labels, clf, NUM_SPLITS):
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True)
    y_pred = np.array([])
    y_true = np.array([])
    feature_imp = np.array([])
    for train_index, test_index in kf.split(resp):
        train_resp = resp[train_index]
        test_resp = resp[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
        feature_imp = np.concatenate((feature_imp, clf.feature_importances_), axis=0)
    f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    return f1, cm, feature_imp


# Settings
TRAINING_DAYS = [3, 6]
Q = 4
DECODER = 'forest'           # bayes, forest, or regression
NUM_SPLITS = 5              # n in n-fold cross validation
ITERATIONS = 100           # how often to decode
METRICS = ['perf_easy', 'threshold', 'bias', 'reaction_time', 'n_trials']

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

# Query days around the day at which criterion is reached
days = (behavior_analysis.BehavioralSummaryByDate
        * subject.Subject
        * subj_crit.proj('subject_uuid')
        & ('training_day between %d and %d' % (TRAINING_DAYS[0], TRAINING_DAYS[1]))).proj(
               'subject_uuid', 'subject_nickname', 'session_date')

# Use dates to query sessions
ses_query = (acquisition.Session).aggr(
        days, from_date='min(session_date)', to_date='max(session_date)')
sessions = (acquisition.Session * ses_query * subject.Subject * subject.SubjectLab * reference.Lab
            & 'date(session_start_time) >= from_date'
            & 'date(session_start_time) <= to_date')


# Create dataframe with behavioral metrics of all mice
behav = pd.DataFrame(columns=['mouse', 'lab', 'perf_easy', 'n_trials',
                              'threshold', 'bias', 'reaction_time', 'days_to_trained'])

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

    # Add results to dataframe
    behav.loc[i, 'mouse'] = nickname
    behav.loc[i, 'lab'] = (sessions & 'subject_nickname = "%s"' % nickname).fetch(
                                                                    'institution_short')[0]
    behav.loc[i, 'perf_easy'] = perf_easy
    behav.loc[i, 'n_trials'] = ntrials_perday
    behav.loc[i, 'threshold'] = fit_result.loc[0, 'threshold']
    behav.loc[i, 'bias'] = fit_result.loc[0, 'bias']
    behav.loc[i, 'reaction_time'] = reaction_time
    behav.loc[i, 'days_to_trained'] = (subj_crit_day
                                       & 'subject_nickname = "%s"' % nickname).fetch(
                                                                             'day_of_crit')[0]

# Drop mice with faulty RT
behav = behav[behav['reaction_time'].notnull()]

# Bin training time
behav['learning_speed'] = pd.qcut(behav['days_to_trained'], q=Q, labels=np.arange(Q))

# Initialize decoders
print('\nDecoding of learning speed..')
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

# Perform decoding of learning speed
labels = list(behav['learning_speed'])
decoding_set = behav[METRICS].values
kf = KFold(n_splits=NUM_SPLITS, shuffle=True)
y_pred = np.zeros(0)
y_true = np.zeros(0)
feature_imp = np.zeros(NUM_SPLITS)
for train_index, test_index in kf.split(decoding_set):
    train_resp = decoding_set[train_index]
    test_resp = decoding_set[test_index]
    clf.fit(train_resp, [labels[j] for j in train_index])
    y_pred = np.append(y_pred, clf.predict(test_resp))
    y_true = np.append(y_true, [labels[j] for j in test_index])
    feature_imp = feature_imp + clf.feature_importances_
feature_imp = feature_imp / [NUM_SPLITS]*NUM_SPLITS
f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
cm = confusion_matrix(y_true, y_pred)
behav['learning_pred'] = y_pred
print('F1 score over chance: %.2f' % (f1-(1/Q)))

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
sns.boxplot(x='learning_speed', y='days_to_trained', data=behav, ax=ax1)
ax1.set(xlabel='Actual learning speed quantile', ylabel='Days to trained', ylim=[0, 80])

sns.boxplot(x='learning_pred', y='days_to_trained', data=behav, ax=ax2)
ax2.set(xlabel='Predicted learning speed quantile', ylabel='Days to trained', ylim=[0, 80])

sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], ax=ax3)
ax3.plot([0, Q], [0, Q], '--w')
ax3.set(xticklabels=np.arange(1, Q+1), yticklabels=np.arange(1, Q+1), ylim=[0, Q], xlim=[0, Q],
        title='Normalized Confusion Matrix', ylabel='Actual speed', xlabel='Predicted speed')


