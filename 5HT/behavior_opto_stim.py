#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from my_functions import load_trials, plot_psychometric, paths, criteria_opto_eids
from oneibl.one import ONE
one = ONE()

# Settings
_, fig_path, _ = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    if subjects.loc[i, 'date_range'] == 'all':
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')
    else:
        eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld',
                          date_range=[subjects.loc[i, 'date_range'][:10], subjects.loc[i, 'date_range'][11:]])
    eids = criteria_opto_eids(eids, max_lapse=0.3, max_bias=0.5, min_trials=300)

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        these_trials = load_trials(eid, laser_stimulation=True)
        these_trials['session'] = ses_count
        trials = trials.append(these_trials, ignore_index=True)
        ses_count = ses_count + 1
    if 'laser_probability' not in trials.columns:
        trials['laser_probability'] = trials['laser_stimulation'].copy()

    sns.set(context='talk', style='ticks', font_scale=1.5)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

    # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=ax1, color='k')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['laser_stimulation'] == 0)
                             & (trials['laser_probability'] != 0.75)], ax=ax1, color='b')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['laser_stimulation'] == 1)
                             & (trials['laser_probability'] != 0.25)], ax=ax1,
                      color='b', linestyle='--')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                             & (trials['laser_stimulation'] == 0)
                             & (trials['laser_probability'] != 0.75)], ax=ax1, color='r')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                             & (trials['laser_stimulation'] == 1)
                             & (trials['laser_probability'] != 0.25)], ax=ax1,
                      color='r', linestyle='--')
    ax1.text(-25, 0.75, '20:80', color='r')
    ax1.text(25, 0.25, '80:20', color='b')
    ax1.set(title='dashed line = opto stim')

    catch_trials = trials[((trials['laser_probability'] == 0.75) & (trials['laser_stimulation'] == 0))
                          | ((trials['laser_probability'] == 0.25) & (trials['laser_stimulation'] == 1))]

    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                 marker='o', label='Stim', color='r', ls='--')
    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.2) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.2) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                 marker='o', label='No stim', color='r')
    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 1) & (trials['laser_probability'] != 0.25)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 1)]['right_choice'].sem()],
                 marker='o', label='Stim', color='b', ls='--')
    ax2.errorbar([0, 1],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].mean(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].mean()],
                 [trials[(trials['probabilityLeft'] == 0.8) & (trials['signed_contrast'] == 0)
                         & (trials['laser_stimulation'] == 0) & (trials['laser_probability'] != 0.75)]['right_choice'].sem(),
                  catch_trials[(catch_trials['probabilityLeft'] == 0.8) & (catch_trials['laser_stimulation'] == 0)]['right_choice'].sem()],
                 marker='o', label='No stim', color='b')
    ax2.set(xticks=[0, 1], xticklabels=['Normal trials', 'Catch trials'], title='0% contrast trials')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_opto_behavior_psycurve' % nickname))