# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from oneibl.one import ONE
one = ONE()

# Settings
FIRST_TRIALS = [10, 12, 14, 16]
FIG_PATH = '/home/guido/Figures/5HT'

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1, index_col=0)
d_types = ['_iblrig_taskSettings.raw',
           'trials.probabilityLeft',
           'trials.contrastLeft',
           'trials.contrastRight',
           'trials.feedbackType',
           'trials.choice']

# Load data
results = pd.DataFrame(columns=['subject', 'condition', 'bias', 'trial'])
for i, nickname in enumerate(sessions.index.values):
    eids = one.search(subject=nickname,
                      date_range=[sessions.loc[nickname, 'Pre-vehicle'],
                                  sessions.loc[nickname, 'Post-vehicle']])
    for j, eid in enumerate(eids):
        d, prob_l, contrast_l, contrast_r, feedback_type, choice = one.load(
                    eid, d_types, dclass_output=False)

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
            first_left = (np.sum(first_choice_l[first_contrast_l == 0] == -1)
                          / np.size(first_choice_l[first_contrast_l == 0]))
            first_right = (np.sum(first_choice_r[first_contrast_r == 0] == -1)
                           / np.size(first_choice_r[first_contrast_r == 0]))
            first_bias[t] = first_right-first_left

            # Add to dataframe
            this_result = pd.DataFrame({'bias': first_bias,
                                        'trial': [str(w) for w in FIRST_TRIALS],
                                        'subject': nickname,
                                        'condition': j})
            results = results.append(this_result, sort=False)

results = results.reset_index()
results['bias'] = results['bias'].astype(float)
results['# trials after block change'] = results['trial']

f, ax1 = plt.subplots(1, 1, figsize=(6, 6))
palette = sns.color_palette('GnBu_d', np.size(FIRST_TRIALS))
sns.lineplot(x='condition', y='bias', hue='trial', data=results,
             ci=68, palette=palette, ax=ax1)
ax1.set(xticks=[0, 1, 2], xticklabels=['Pre-\nvehicle', '5HT2a\nantagonist', 'Post-\nvehicle'],
        xlabel='', ylabel='Bias', ylim=[-0.1, 0.6])
legend = ax1.legend(loc=[0.05, 0.62], frameon=False, fontsize=12)
legend.texts[0].set_text('Trials')
legend.texts[0].set_position((0.1, 0.1))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40)
sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine(trim=True)
plt.tight_layout(pad=2)

plt.savefig(join(FIG_PATH, '5HT2a_block_bias.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, '5HT2a_block_bias.png'), dpi=300)
