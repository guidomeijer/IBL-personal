# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, expanduser
import seaborn as sns
from oneibl.one import ONE
one = ONE()

# Settings
FIRST_TRIALS = 15
FIG_PATH = join(expanduser('~'), 'Figures', '5HT')
d_types = ['_iblrig_taskSettings.raw',
           'trials.probabilityLeft',
           'trials.contrastLeft',
           'trials.contrastRight',
           'trials.feedbackType',
           'trials.choice']

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1, index_col=0)

# Load data
results = pd.DataFrame(columns=['subject', 'condition', 'bias', 'first_bias', 'n_trials',
                                'perf_easy'])
for i, nickname in enumerate(sessions.index.values):
    eids = one.search(subject=nickname,
                      date_range=[sessions.loc[nickname, 'Pre-vehicle'],
                                  sessions.loc[nickname, 'Post-vehicle']])

    if len(eids) > 3:
        eids = eids[0:3]

    for j, eid in enumerate(eids):
        d, prob_l, contrast_l, contrast_r, feedback_type, choice = one.load(
                    eid, d_types, dclass_output=False)

        # Calculate bias
        left = (np.sum(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.2)] == -1)
                / np.size(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.2)]))
        right = (np.sum(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.8)] == -1)
                 / np.size(choice[((contrast_l == 0) | (contrast_r == 0)) & (prob_l == 0.8)]))

        # Get the first trials after block switch
        left_blocks = np.where(np.diff(prob_l) > 0.5)[0]
        first_contrast_l = np.zeros(0)
        first_choice_l = np.zeros(0)
        for k, ind in enumerate(left_blocks):
            first_contrast_l = np.append(first_contrast_l, contrast_l[ind+1:ind+FIRST_TRIALS+1])
            first_choice_l = np.append(first_choice_l, choice[ind+1:ind+FIRST_TRIALS+1])

        right_blocks = np.where(np.diff(prob_l) < -0.5)[0]
        first_contrast_r = np.zeros(0)
        first_choice_r = np.zeros(0)
        for k, ind in enumerate(right_blocks):
            first_contrast_r = np.append(first_contrast_r, contrast_r[ind+1:ind+FIRST_TRIALS+1])
            first_choice_r = np.append(first_choice_r, choice[ind+1:ind+FIRST_TRIALS+1])

        # Calculate bias per contrast for first trials
        first_left = (np.sum(first_choice_l[first_contrast_l == 0] == -1)
                / np.size(first_choice_l[first_contrast_l == 0]))
        first_right = (np.sum(first_choice_r[first_contrast_r == 0] == -1)
                 / np.size(first_choice_r[first_contrast_r == 0]))
        first_bias = first_right-first_left

        # Calculate performance
        perf_easy = (np.sum(feedback_type[((contrast_l >= 0.5) | (contrast_r >= 0.5))] == 1)
                     / np.size(feedback_type[((contrast_l >= 0.5) | (contrast_r >= 0.5))]))*100

        # Add to dataframe
        results.loc[results.shape[0]+1, 'subject'] = nickname
        results.loc[results.shape[0], 'bias'] = left-right
        results.loc[results.shape[0], 'first_bias'] = first_bias
        results.loc[results.shape[0], 'n_trials'] = np.size(choice)
        results.loc[results.shape[0], 'perf_easy'] = perf_easy
        results.loc[results.shape[0], 'condition'] = j

results[['bias', 'first_bias', 'n_trials', 'perf_easy']] = results[
            ['bias', 'first_bias', 'n_trials', 'perf_easy']].astype(float)
results['subject'] = results['subject'].astype(str)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
sns.lineplot(x='condition', y='bias', hue='subject', data=results, lw=3, ax=ax1)
ax1.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Bias', title='Overall bias strenght',
        ylim=[0, 0.6])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(fontsize=10, frameon=False, handles=handles[1:], labels=labels[1:])
sns.lineplot(x='condition', y='first_bias', hue='subject', data=results,
             legend=False, lw=3, ax=ax2)
ax2.set(xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a block', 'Post-vehicle'],
        xlabel='', ylabel='Bias', title='Bias in first %d trials after switch' % FIRST_TRIALS,
        ylim=[-0.1, 0.5])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40)
sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine(trim=True)
plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, '5HT2a_bias.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, '5HT2a_bias.png'), dpi=300)
