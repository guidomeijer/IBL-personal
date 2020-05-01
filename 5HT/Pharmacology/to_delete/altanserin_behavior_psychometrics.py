#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 10:36:23 2020

@author: guido
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from dj_tools import dj2pandas, fit_psychfunc, plot_psychometric
from psychometric_functions import plot_psychometric_data
from ibl_pipeline import subject, acquisition, reference, behavior

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1)
sessions = sessions.loc[sessions['Week'] == 1]

colors = sns.color_palette(n_colors=3)

f, ax = plt.subplots(1, sessions.shape[0] + 1, figsize=(20, 5))
sns.set(style="ticks", context="paper", font_scale=1.5)
fit_results = pd.DataFrame()
for j, condition in enumerate(['Pre-vehicle', 'Drug', 'Post-vehicle']):
    over_mice = pd.DataFrame()
    for i in range(sessions.shape[0]):

        # Load in data
        ses_query = (acquisition.Session * subject.Subject
                     & 'subject_nickname = "%s"' % sessions.loc[i, 'Nickname']
                     & 'date(session_start_time) = "%s"' % sessions.loc[i, condition]
                     & 'task_protocol LIKE "%biased%"')
        assert len(ses_query) == 1
        trials = (ses_query * behavior.TrialSet.Trial).fetch(format='frame')
        trials = trials.reset_index()

        # Fit psychometric data
        fit_df = dj2pandas(trials)
        result = fit_psychfunc(fit_df)
        result['condition'] = condition
        result['subject'] = sessions.loc[i, 'Nickname']
        fit_results = fit_results.append(result)
        plot_psychometric(fit_df['signed_contrast'], fit_df['choice_right'],
                          fit_df['subject_nickname'], ax=ax[i], color=colors[j])
        ax[i].set(xlabel='Signed contrast (%)', ylabel='Rightward responses')

        # Add to dataframe
        over_mice = over_mice.append(fit_df)

    plot_psychometric(over_mice['signed_contrast'], over_mice['choice_right'],
                      over_mice['subject_nickname'], ax=ax[-1], color=colors[j],
                      label=condition)
ax[-1].legend(['Pre-vehicle', '_', '_', '_', 'Drug', '_', '_', '_', 'Post-vehicle'], frameon=False)
sns.despine(trim=True)
plt.tight_layout()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
sns.set(style="ticks", context="paper", font_scale=2)
sns.lineplot(x='condition', y='bias', data=fit_results, estimator=None, units='subject',
             color='gray', ax=ax1)
sns.lineplot(x='condition', y='bias', data=fit_results, ci=68, ax=ax1)
ax1.set(ylim=[0, 10])

sns.set(style="ticks", context="paper", font_scale=2)
sns.lineplot(x='condition', y='threshold', data=fit_results, estimator=None, units='subject',
             color='gray', ax=ax2)
sns.lineplot(x='condition', y='threshold', data=fit_results, ci=68, ax=ax2)



