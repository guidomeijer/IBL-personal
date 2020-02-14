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
FIRST_TRIALS = [10, 15, 20]
FIG_PATH = join(expanduser('~'), 'Figures', '5HT')
MAX_CONTRAST = 0.1

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

        first_bias = np.zeros(np.size(FIRST_TRIALS))
        for t, trial in enumerate(FIRST_TRIALS):

            # Get the first trials after block switch
            left_blocks = np.where(np.diff(prob_l) > 0.5)[0]
            first_contrast_l = np.zeros(0)
            first_choice_l = np.zeros(0)
            for k, ind in enumerate(left_blocks):
                first_contrast_l = np.append(first_contrast_l, contrast_l[ind+1:ind+trial+1])
                first_choice_l = np.append(first_choice_l, choice[ind+1:ind+trial+1])

            right_blocks = np.where(np.diff(prob_l) < -0.5)[0]
            first_contrast_r = np.zeros(0)
            first_choice_r = np.zeros(0)
            for k, ind in enumerate(right_blocks):
                first_contrast_r = np.append(first_contrast_r, contrast_r[ind+1:ind+trial+1])
                first_choice_r = np.append(first_choice_r, choice[ind+1:ind+trial+1])

            # Calculate bias per contrast for first trials
            first_left = (np.sum(first_choice_l[first_contrast_l < MAX_CONTRAST] == -1)
                          / np.size(first_choice_l[first_contrast_l < MAX_CONTRAST]))
            first_right = (np.sum(first_choice_r[first_contrast_r < MAX_CONTRAST] == -1)
                           / np.size(first_choice_r[first_contrast_r < MAX_CONTRAST]))
            first_bias[t] = first_right-first_left

        # Add to dataframe
        this_result = pd.DataFrame({'bias': first_bias,
                                    'trial': [str(w) for w in FIRST_TRIALS],
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
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
palette = sns.color_palette('GnBu_d', np.size(FIRST_TRIALS))

sns.lineplot(x='condition', y='bias', units='subject', estimator=None, hue='subject', sort=False,
             data=results[(results['week'] == 1) & (results['trial'] == str(FIRST_TRIALS[0]))],
             ax=ax1)
ax1.set(xlabel='', ylabel='Bias', ylim=[-0.1, 0.6])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.lineplot(x='condition', y='bias', units='subject', estimator=None, hue='subject', sort=False,
             data=results[(results['week'] == 2) & (results['trial'] == str(FIRST_TRIALS[0]))],
             ax=ax2)
ax2.set(xlabel='', ylabel='Bias', ylim=[-0.1, 0.6])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)

sns.set(context='paper', font_scale=1.5, style='ticks')
plt.tight_layout(pad=2)

f, ax1 = plt.subplots(1, 1, figsize=(6, 6))
sns.lineplot(x='condition', y='bias_rel', hue='trial', data=results,
             ci=68, palette=palette, sort=False, ax=ax1)
ax1.set(xlabel='', ylabel='Bias', title='5HT2a antagonist (altanserin)')
legend = ax1.legend(loc=[0.05, 0.1], frameon=False, fontsize=12)
legend.texts[0].set_text('First trials')
legend.texts[0].set_position((0.1, 0.1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)

sns.set(context='paper', font_scale=1.5, style='ticks')
plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, '5HT2a_block_bias.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, '5HT2a_block_bias.png'), dpi=300)
