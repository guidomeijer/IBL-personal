# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
from os.path import join, expanduser
import matplotlib.pyplot as plt
import seaborn as sns
from oneibl.one import ONE
one = ONE()

# Settings
FIG_PATH = join(expanduser('~'), 'Figures', '5HT')

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1)
d_types = ['_iblrig_taskSettings.raw',
           'trials.probabilityLeft',
           'trials.contrastLeft',
           'trials.contrastRight',
           'trials.feedbackType',
           'trials.choice']

# Load data
results = pd.DataFrame()
for i in range(sessions.shape[0]):
    for j, day in enumerate(['Pre-vehicle', 'Drug', 'Post-vehicle']):
        eid = one.search(subject=sessions.loc[i, 'Nickname'],
                         date_range=[sessions.loc[i, day], sessions.loc[i, day]],
                         task_protocol='_iblrig_tasks_biasedChoiceWorld')
        assert len(eid) == 1
        d, prob_l, contrast_l, contrast_r, feedback_type, choice = one.load(eid[0], d_types,
                                                                            dclass_output=False)

        right_block = (np.sum(choice[(prob_l < 0.45) & ((contrast_l == 0)
                                                        | (contrast_r == 0))] == 1)
                       / choice[(prob_l < 0.45) & ((contrast_l == 0)
                                                   | (contrast_r == 0))].shape[0])
        left_block = (np.sum(choice[(prob_l > 0.55) & ((contrast_l == 0)
                                                       | (contrast_r == 0))] == 1)
                      / choice[(prob_l > 0.55) & ((contrast_l == 0)
                                                  | (contrast_r == 0))].shape[0])
        bias = left_block - right_block

        # Add to dataframe
        this_result = pd.DataFrame(index=[0], data={'bias': bias,
                                                    'subject': sessions.loc[i, 'Nickname'],
                                                    'condition': day,
                                                    'week': sessions.loc[i, 'Week']})
        results = results.append(this_result, sort=False)

results = results.reset_index()
results['bias'] = results['bias'].astype(float)

# Get bias normalized to pre-vehicle
results.loc[results['condition'] == 'Pre-vehicle', 'bias_rel'] = (
                        results.loc[results['condition'] == 'Pre-vehicle', 'bias'].values
                        / results.loc[results['condition'] == 'Pre-vehicle', 'bias'].values)
results.loc[results['condition'] == 'Drug', 'bias_rel'] = (
                        results.loc[results['condition'] == 'Drug', 'bias'].values
                        / results.loc[results['condition'] == 'Pre-vehicle', 'bias'].values)
results.loc[results['condition'] == 'Post-vehicle', 'bias_rel'] = (
                        results.loc[results['condition'] == 'Post-vehicle', 'bias'].values
                        / results.loc[results['condition'] == 'Pre-vehicle', 'bias'].values)

# Plot results
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

sns.lineplot(x='condition', y='bias', units='subject', estimator=None, hue='subject', sort=False,
             data=results[results['week'] == 1], ax=ax1)
ax1.set(xlabel='', ylabel='Bias', ylim=[-0.1, 0.6])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.lineplot(x='condition', y='bias', units='subject', estimator=None, hue='subject', sort=False,
             data=results[results['week'] == 2], ax=ax2)
ax2.set(xlabel='', ylabel='Bias', ylim=[-0.1, 0.6])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

sns.lineplot(x='condition', y='bias', units='subject', estimator=None, hue='subject', sort=False,
             data=results[results['week'] == 3], ax=ax3)
ax3.set(xlabel='', ylabel='Bias', ylim=[-0.1, 0.6])
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=40)

sns.set(context='paper', font_scale=1.5, style='ticks')
plt.tight_layout(pad=2)
