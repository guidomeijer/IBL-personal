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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, LeaveOneOut

# Settings
DAYS = np.arange(1, 9)
NUM_SPLITS = 1              # n in n-fold cross validation (1 for leave one out)
METRICS = ['performance', 'mean_rt', 'n_trials_date']
FIG_PATH = '/home/guido/Figures/Behavior/LearningSpeedPrediction'
DATA_PATH = '/home/guido/Repositories/IBL-personal/Behavior/data'

F1_score = np.empty(DAYS.shape[0])
for i, day in enumerate(DAYS):
    # Get dataframe with behavioral data
    behav = pd.read_csv(join(DATA_PATH, 'training_day_%d.csv' % day))

    # Perform multiple linear regression
    # reg = LogisticRegression(solver='newton-cg', multi_class='auto')
    reg = LinearRegression()
    X = behav[METRICS].values
    Y = behav['day_of_crit'].values
    if NUM_SPLITS == 1:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    y_pred = np.zeros(0)
    y_true = np.zeros(0)
    for train_index, test_index in cv.split(X):
        reg.fit(X[train_index], Y[train_index])
        y_pred = np.append(y_pred, reg.predict(X[test_index]))
        y_true = np.append(y_true, Y[test_index])


    f, ax1 = plt.subplots(1, 1, figsize=(5.5, 5))
    # sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], vmin=0.15, vmax=0.5, ax=ax1)
    sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], ax=ax1)
    ax1.plot([0, Q], [0, Q], '--w')
    ax1.set(xticklabels=np.arange(1, Q+1), yticklabels=np.arange(1, Q+1), ylim=[0, Q], xlim=[0, Q],
            title='Day %d' % day, ylabel='Actual learning speed quantile',
            xlabel='Predicted learning speed quantile')
    plt.savefig(join(FIG_PATH, 'confusion_matrix_day%d' % day), dpi=300)
    plt.close(f)

f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
ax1.plot(DAYS, F1_score)
ax1.plot([DAYS[0], DAYS[-1]], [1/Q, 1/Q], color=[0.6, 0.6, 0.6], linestyle='--')
ax1.set(ylabel='Classification performance (F1 score)',
        xlabel='Classifier trained on behavior from day')
plt.savefig(join(FIG_PATH, 'learning_speed_classification'), dpi=300)
