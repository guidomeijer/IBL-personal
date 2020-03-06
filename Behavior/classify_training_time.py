#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:28:14 2019

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
import datajoint as dj
from ibl_pipeline import subject, acquisition
from ibl_pipeline.analyses import behavior as behavior_analysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut
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
Q = 4
DAYS = np.arange(1, 11)
DECODER = 'forest'           # bayes, forest, or regression
NUM_SPLITS = 1              # n in n-fold cross validation (1 for leave one out)
METRICS = ['performance', 'mean_rt', 'n_trials_date']
FIG_PATH = '/home/guido/Figures/Behavior/LearningSpeedPrediction'

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
rt = behavior_analysis.ReactionTime.proj('reaction_time', session_date='DATE(session_start_time)')

# Initialize decoders
if DECODER == 'forest':
    clf = RandomForestClassifier(n_estimators=100)
elif DECODER == 'bayes':
    clf = GaussianNB()
elif DECODER == 'regression':
    clf = LogisticRegression(solver='liblinear', multi_class='auto')
else:
    raise Exception('DECODER must be forest, bayes or regression')

results = pd.DataFrame(columns=['day', 'F1'] + METRICS)
for i, day in enumerate(DAYS):
    print('Decoding of learning speed from day %d' % day)

    # Get dataframe with behavioral data
    behav = (subj_crit_day * behavior_analysis.BehavioralSummaryByDate * rt
             & 'training_day="%d"' % day).fetch(format='frame')
    behav = behav.reset_index()
    behav['mean_rt'] = [np.mean(i) for i in behav['reaction_time']]

    # Bin training time
    behav['learning_speed'] = pd.qcut(behav['day_of_crit'], q=Q, labels=np.arange(Q))

    # Drop nans in rt
    behav = behav[behav['mean_rt'].notnull()]

    # Perform decoding of learning speed
    labels = list(behav['learning_speed'])
    decoding_set = behav[METRICS].values
    if NUM_SPLITS == 1:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    y_pred = np.zeros(0)
    y_true = np.zeros(0)
    feature_imp = np.zeros(len(METRICS))
    for train_index, test_index in cv.split(decoding_set):
        train_resp = decoding_set[train_index]
        test_resp = decoding_set[test_index]
        clf.fit(train_resp, [labels[j] for j in train_index])
        y_pred = np.append(y_pred, clf.predict(test_resp))
        y_true = np.append(y_true, [labels[j] for j in test_index])
        feature_imp = feature_imp + clf.feature_importances_
    if NUM_SPLITS == 1:
        feature_imp = feature_imp / [decoding_set.shape[0]]*decoding_set.shape[1]
    else:
        feature_imp = feature_imp / [NUM_SPLITS]*decoding_set.shape[1]
    f1 = f1_score(y_true, y_pred, labels=np.unique(labels), average='micro')
    cm = confusion_matrix(y_true, y_pred)
    behav['learning_pred'] = y_pred
    results.loc[i] = np.append([day, f1], feature_imp)

    f, ax1 = plt.subplots(1, 1, figsize=(5.5, 5))
    # sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], vmin=0.15, vmax=0.5, ax=ax1)
    sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], ax=ax1)
    ax1.plot([0, Q], [0, Q], '--w')
    ax1.set(xticklabels=np.arange(1, Q+1), yticklabels=np.arange(1, Q+1), ylim=[0, Q], xlim=[0, Q],
            title='Day %d' % day, ylabel='Actual learning speed quantile',
            xlabel='Predicted learning speed quantile')
    plt.savefig(join(FIG_PATH, 'confusion_matrix_day%d' % day), dpi=300)
    plt.close(f)

# %%

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(results['day'], results['F1'])
ax1.plot([DAYS[0], DAYS[-1]], [1/Q, 1/Q], color=[0.6, 0.6, 0.6], linestyle='--')
ax1.set(ylabel='Classification performance (F1 score)',
        xlabel='Classifier trained on behavior from day')

ax2.plot(results['day'], results['performance'])
ax2.plot(results['day'], results['mean_rt'])
ax2.plot(results['day'], results['n_trials_date'])
ax2.legend(['Performance', 'RT', '# trials'])
ax2.set(ylabel='Importance of predictor',
        xlabel='Classifier trained on behavior from day')

plt.savefig(join(FIG_PATH, 'learning_speed_classification'), dpi=300)
