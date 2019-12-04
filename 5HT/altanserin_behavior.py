# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from oneibl.one import ONE
one = ONE()

# Settings
FIRST_TRIALS = 10
d_types = ['_iblrig_taskSettings.raw',
           'trials.probabilityLeft',
           'trials.contrastLeft',
           'trials.contrastRight',
           'trials.feedbackType',
           'trials.choice']

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1, index_col=0)

# Load data
results = pd.DataFrame(columns=['subject', 'bias', 'first_bias', 'condition'])
for i, nickname in enumerate(sessions.index.values):
    eids = one.search(subject=nickname,
                      date_range=[sessions.loc[nickname, 'Pre-vehicle'],
                                  sessions.loc[nickname, 'Post-vehicle']])
    for j, eid in enumerate(eids):
        d, prob_l, contrast_l, contrast_r, feedback_type, choice = one.load(
                    eid, d_types, dclass_output=False)

        # Calculate bias
        left = (np.sum(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.2)] == -1)
                / np.size(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.2)]))
        right = (np.sum(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.8)] == -1)
                 / np.size(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.8)]))

        # Calculate bias in first trials after block switch
        left_blocks = np.where(np.diff(prob_l) > 0.5)[0]
        bias_left = np.zeros(0)
        for k, ind in enumerate(left_blocks):
            this_choice = choice[ind+1:ind+FIRST_TRIALS+1]
            this_contr_l = contrast_l[ind+1:ind+FIRST_TRIALS+1]
            this_contr_r = contrast_r[ind+1:ind+FIRST_TRIALS+1]
            bias = (np.sum(this_choice[((this_contr_l == 0) | (this_contr_r == 0))] == -1)
                    / np.size(this_choice[((this_contr_l == 0) | (this_contr_r == 0))]))
            bias_left = np.append(bias_left, bias)

        right_blocks = np.where(np.diff(prob_l) < -0.5)[0]
        bias_right = np.zeros(0)
        for k, ind in enumerate(right_blocks):
            this_choice = choice[ind+1:ind+FIRST_TRIALS+1]
            this_contr_l = contrast_l[ind+1:ind+FIRST_TRIALS+1]
            this_contr_r = contrast_r[ind+1:ind+FIRST_TRIALS+1]
            bias = (np.sum(this_choice[((this_contr_l == 0) | (this_contr_r == 0))] == -1)
                    / np.size(this_choice[((this_contr_l == 0) | (this_contr_r == 0))]))
            bias_right = np.append(bias_right, bias)

        first_bias = (np.mean(bias_right[np.isnan(bias_right) == 0])
                      - np.mean(bias_left[np.isnan(bias_left) == 0]))

        # Add to dataframe
        results.loc[results.shape[0]+1, 'subject'] = nickname
        results.loc[results.shape[0], 'bias'] = left-right
        # results.loc[results.shape[0], 'first_bias'] = np.mean(all_bias[np.isnan(all_bias) == 0])
        results.loc[results.shape[0], 'first_bias'] = first_bias
        results.loc[results.shape[0], 'condition'] = j

results[['bias', 'first_bias']] = results[['bias', 'first_bias']].astype(float)
results['subject'] = results['subject'].astype(str)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
sns.lineplot(x='condition', y='bias', hue='subject', data=results, ax=ax1)
ax1.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Bias', title='Overall bias strenght',
        ylim=[0, 0.6])

sns.lineplot(x='condition', y='first_bias', hue='subject', data=results, ax=ax2)
ax2.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Bias', title='Bias in first %d trials after switch' % FIRST_TRIALS,
        ylim=[0, 0.6])

plt.tight_layout(pad=2)