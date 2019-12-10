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
from fit_psytrack import fit_model, plot_psytrack

# Settings
TRIAL_WIN = [0, 15]
FIG_PATH = join(expanduser('~'), 'Figures', '5HT')

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1, index_col=0)

# Load data
results = pd.DataFrame(columns=['subject', 'condition', 'slope_left', 'slope_right', 'slope_abs'])
for i, nickname in enumerate(sessions.index.values):
    for j, condition in enumerate(sessions.columns.values):

        # Fit model
        model, prob_l, _ = fit_model(nickname, [sessions.loc[nickname, condition],
                                                sessions.loc[nickname, condition]])

        # Fit slopes
        ax = plot_psytrack(model, prob_l, False)
        ax.set(title='%s; %s' % (nickname, condition))
        trial_vec = np.append(np.arange(-TRIAL_WIN[0], 0), np.arange(1, TRIAL_WIN[1]+1))
        left_blocks = np.where(np.diff(prob_l) > 0.2)[0]
        left_slope = np.zeros(np.size(left_blocks))
        for l, ind in enumerate(left_blocks):
            poly = np.polyfit(trial_vec, model[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]], 1)
            left_slope[l] = np.rad2deg(np.arctan(poly[0]))
            ax.plot(np.arange(ind-TRIAL_WIN[0], ind+TRIAL_WIN[1]),
                    np.polyval(poly, trial_vec), 'g', lw=3)

        right_blocks = np.where(np.diff(prob_l) < -0.2)[0]
        right_slope = np.zeros(np.size(right_blocks))
        for l, ind in enumerate(right_blocks):
            poly = np.polyfit(trial_vec, model[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]], 1)
            right_slope[l] = np.rad2deg(np.arctan(poly[0]))
            ax.plot(np.arange(ind-TRIAL_WIN[0], ind+TRIAL_WIN[1]),
                    np.polyval(poly, trial_vec), 'g', lw=3)
        results = results.append(pd.DataFrame(
                                    index=[0], data={'subject': nickname, 'condition': j,
                                                     'slope_left': np.mean(left_slope),
                                                     'slope_right': np.mean(right_slope),
                                                     'slope_abs': np.mean(np.abs(np.append(
                                                                     left_slope, right_slope)))}))
        plt.savefig(join(FIG_PATH, 'psytrack_fit_%s_%s.png' % (nickname, condition)), dpi=300)
        plt.savefig(join(FIG_PATH, 'psytrack_fit_%s_%s.pdf' % (nickname, condition)), dpi=300)

f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x='condition', y='slope_abs', data=results, style='subject',
             legend=False, lw=3, markers=['o', 'o'], dashes=['', ''], ax=ax1)
ax1.set(ylabel='Speed of bias switch (absolute slope)',
        xticks=[0, 1, 2], xticklabels=['Pre-vehicle', '5HT2a antagonist', 'Post-vehicle'],
        ylim=[0, 12])
sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine(trim=True)

plt.savefig(join(FIG_PATH, 'altanserin_slope_model.png'), dpi=300)
plt.savefig(join(FIG_PATH, 'altanserin_slope_model.pdf'), dpi=300)
