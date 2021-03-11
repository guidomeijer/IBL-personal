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
from my_functions import paths, figure_style, get_full_region_name, get_parent_region_name

# Settings
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Ephys', 'Decoding')
FULL_NAME = True
PARENT_REGIONS = False

# %% Plot
# Load in data
method_a = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'linear-regression',
                            'prior-stimside_pseudo_kfold-interleaved_aligned-behavior_all_cells_allen-atlas.p'))
method_b = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'linear-regression',
                            'prior-prevaction_pseudo_kfold-interleaved_aligned-behavior_all_cells_allen-atlas.p'))

# Get decoding performance over chance
method_a['r_over_chance'] = (method_a['r_prior'] - method_a['r_prior_null'])
method_b['r_over_chance'] = (method_b['r_prior'] - method_b['r_prior_null'])

# Get decoding average per region
method_a_avg = method_a.groupby('region').mean()['r_prior']
method_b_avg = method_b.groupby('region').mean()['r_prior']

# Get regions that are in both
method_a_avg = method_a_avg[method_a_avg.index.isin(method_b_avg.index)]
method_b_avg = method_b_avg[method_b_avg.index.isin(method_a_avg.index)]

# %%
figure_style(font_scale=1.8)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

ax1.scatter(method_a_avg, method_b_avg)
m, b = np.polyfit(method_a_avg, method_b_avg, 1)
ax1.plot(np.arange(-0.2, 0.6, 0.01), m*np.arange(-0.2, 0.6, 0.01) + b, lw=2, color='k')
r, p = pearsonr(method_a_avg, method_b_avg)
#ax1.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
ax1.set(xlabel='Exponential smoothing stimulus sides (r)',
        ylabel='Exponential smoothing actions (r)',
        xlim=[-0.2, 0.6], ylim=[-0.2, 0.6],
        title='r=%.2f, p=%.2f' % (r, p))

ax2.errorbar([0, 1], [method_a_avg.mean(), method_b_avg.mean()],
             [method_a_avg.sem(), method_b_avg.sem()], lw=2)
_, p = ttest_rel(method_a_avg, method_b_avg)
ax2.set(xticks=[0, 1], xticklabels=['Stim side', 'Actions'],
        ylabel='Decoding performance (r)', title='p = %.3f' % p)

plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_stimside_vs_actions'))
