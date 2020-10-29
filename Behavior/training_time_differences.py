#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantify the variability of the time to trained over labs.

@author: Guido Meijer
16 Jan 2020
"""
from os.path import join

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

from paper_behavior_functions import (query_subjects, seaborn_style, institution_map,
                                      group_colors, figpath, datapath, EXAMPLE_MOUSE,
                                      FIGURE_HEIGHT, FIGURE_WIDTH)
from ibl_pipeline.analyses import behavior as behavior_analysis

# Settings
fig_path = figpath()

# Query sessions
use_subjects = query_subjects()
ses = ((use_subjects * behavior_analysis.SessionTrainingStatus * behavior_analysis.PsychResults
        & 'training_status = "in_training" OR training_status = "untrainable"')
       .proj('subject_nickname', 'n_trials_stim', 'institution_short', 'lab_name', 'sex')
       .fetch(format='frame')
       .dropna()
       .reset_index())
ses['n_trials'] = [sum(i) for i in ses['n_trials_stim']]

# Construct dataframe
training_time = pd.DataFrame(columns=['sessions'], data=ses.groupby('subject_nickname').size())
training_time['trials'] = ses.groupby('subject_nickname').sum()
training_time['sex'] = ses.groupby('subject_nickname')['sex'].apply(list).str[0]
training_time['lab'] = ses.groupby('subject_nickname')['lab_name'].apply(list).str[0]
training_time.loc[training_time['sex'] == 'M', 'sex'] = 'Male'
training_time.loc[training_time['sex'] == 'F', 'sex'] = 'Female'


# %% PLOT

# Plot labs with differences against each other
sns.set(style="ticks", context="paper", font_scale=1.5)
# sns.set(style="ticks", context="talk", font_scale=1)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5), dpi=150, sharey=True)

training_time.loc[training_time['lab'].isin(['churchlandlab', 'wittenlab', 'angelakilab']),
                  'light_cycle'] = 'Inverted'
training_time.loc[training_time['lab'].isin(['mainenlab', 'cortexlab', 'danlab', 'zadorlab']),
                  'light_cycle'] = 'Non-inverted'
p = stats.ttest_ind(training_time.loc[training_time['light_cycle'] == 'Inverted', 'sessions'],
                    training_time.loc[training_time['light_cycle'] == 'Non-inverted',
                                      'sessions'])[1]
sns.boxplot(x='light_cycle', y='sessions', data=training_time,
            color=[0.6, 0.6, 0.6], ax=ax1)
ax1.set(title='Light cycle (p = %.3f)' % p, ylabel='Days to trained', xlabel='', ylim=[0, 60])

training_time.loc[training_time['lab'].isin(['churchlandlab', 'wittenlab', 'angelakilab',
                                             'danlab', 'zadorlab']),
                  'provider'] = 'Charles-River\n(Europe)'
training_time.loc[training_time['lab'].isin(['mainenlab', 'cortexlab', 'mrsicflogellab',
                                             'hoferlab']),
                  'provider'] = 'Jax\n(US)'
p = stats.ttest_ind(training_time.loc[training_time['provider'] == 'Charles-River\n(Europe)',
                                      'sessions'],
                    training_time.loc[training_time['provider'] == 'Jax\n(US)',
                                      'sessions'])[1]
sns.boxplot(x='provider', y='sessions', data=training_time, color=[0.6, 0.6, 0.6], ax=ax2)
ax2.set(title='Provider (p = %.3f)' % p, ylabel='', xlabel='')

p = stats.ttest_ind(training_time.loc[training_time['sex'] == 'Male', 'sessions'],
                    training_time.loc[training_time['sex'] == 'Female', 'sessions'])[1]
sns.boxplot(x='sex', y='sessions', data=training_time, color=[0.6, 0.6, 0.6], ax=ax3)
ax3.set(title='Sex (p = %.3f)' % p, ylabel='', xlabel='')

plt.tight_layout(pad=2)
sns.despine(trim=True)




