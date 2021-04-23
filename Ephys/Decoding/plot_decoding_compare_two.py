#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode left/right block identity from all brain regions
@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, pearsonr
from my_functions import paths, figure_style

# Settings
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Ephys', 'Decoding')
FULL_NAME = True
PARENT_REGIONS = False
LABEL_A = '-600 to -100 ms'
LABEL_B = '-200 to 0 ms'
TITLE_STR = 'Comparison of timewindows'
SAVE_STR = 'timewindow_comparison'

# %% Plot
# Load in data
method_a = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'linear-regression',
                'prior-prevaction_other-trials_kfold_aligned-behavior_all_cells_beryl-atlas_600-100.p'))
method_b = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'linear-regression',
                'prior-prevaction_other-trials_kfold_aligned-behavior_all_cells_beryl-atlas_200-0.p'))

# Get decoding performance over chance
method_a['r_over_chance'] = (method_a['r'] - method_a['r_null'])
method_b['r_over_chance'] = (method_b['r'] - method_b['r_null'])

# Get decoding average per region
method_a_avg = method_a.groupby('region').mean()['r']
method_b_avg = method_b.groupby('region').mean()['r']
method_a_avg = method_a_avg[method_a_avg.index.isin(method_b_avg.index)]
method_b_avg = method_b_avg[method_b_avg.index.isin(method_a_avg.index)]

method_a_over = method_a.groupby('region').mean()['r_over_chance']
method_b_over = method_b.groupby('region').mean()['r_over_chance']
method_a_over = method_a_over[method_a_over.index.isin(method_b_over.index)]
method_b_over = method_b_over[method_b_over.index.isin(method_a_over.index)]

# %%
figure_style(font_scale=1.7)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
f.suptitle(TITLE_STR, fontsize=21)

ax1.scatter(method_a_avg, method_b_avg)
m, b = np.polyfit(method_a_avg, method_b_avg, 1)
ax1.plot(np.arange(-0.2, 0.6, 0.01), m*np.arange(-0.2, 0.6, 0.01) + b, lw=2, color='k')
r, p = pearsonr(method_a_avg, method_b_avg)
#ax1.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
ax1.set(xlabel='%s (r)' % LABEL_A, ylabel='%s (r)' % LABEL_B,
        xlim=[-0.2, 0.6], ylim=[-0.2, 0.6],
        title='Decoding performance\n(Pearson, r=%.2f, p=%.2f)' % (r, p))

ax2.errorbar([0, 1], [method_a_avg.mean(), method_b_avg.mean()],
             [method_a_avg.sem(), method_b_avg.sem()], lw=2)
_, p = ttest_rel(method_a_avg, method_b_avg)
ax2.set(xticks=[0, 1], xticklabels=[LABEL_A, LABEL_B],
        ylabel='r', title='Decoding performance\n(paired t-test, p = %.2f)' % p)

ax3.scatter(method_a_over, method_b_over)
m, b = np.polyfit(method_a_over, method_b_over, 1)
ax3.plot(np.arange(-0.2, 0.6, 0.01), m*np.arange(-0.2, 0.6, 0.01) + b, lw=2, color='k')
r, p = pearsonr(method_a_over, method_b_over)
#ax1.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
ax3.set(xlabel='%s (r)' % LABEL_A, ylabel='%s (r)' % LABEL_B,
        xlim=[-0.2, 0.6], ylim=[-0.2, 0.6],
        title='Decoding improvement over chance\n(Pearson, r=%.2f, p=%.2f)' % (r, p))

ax4.errorbar([0, 1], [method_a_over.mean(), method_b_over.mean()],
             [method_a_over.sem(), method_b_over.sem()], lw=2)
_, p = ttest_rel(method_a_over, method_b_over)
ax4.set(xticks=[0, 1], xticklabels=[LABEL_A, LABEL_B],
        ylabel='r', title='Decoding improvement over chance\n(paired t-test, p = %.2f)' % p)

plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_%s' % SAVE_STR))
