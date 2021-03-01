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
from scipy.stats import ttest_rel
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
                            'prior_pseudo_kfold-interleaved_aligned-behavior_all_cells.p'))

# Get decoding performance over chance
method_a['r_over_chance'] = (method_a['r_prior'] - method_a['r_prior_null'])
method_b['r_over_chance'] = (method_b['r_infer'] - method_b['r_infer_null'])

# Get decoding average per region
method_a_avg = method_a.groupby('region').mean()['r_over_chance']
method_b_avg = method_b.groupby('region').mean()['r_over_chance']

# Get regions that are in both
method_a_avg = method_a_avg[method_a_avg.index.isin(method_b_avg.index)]
method_b_avg = method_b_avg[method_b_avg.index.isin(method_a_avg.index)]

# %%
figure_style(font_scale=1.8)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=150)

ax1.scatter(method_a_avg, method_b_avg)
ax1.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
ax1.set(xlabel='Exponential smoothing model decoding (r over chance)',
        ylabel='Optimal model decoding (r over chance)',
        xlim=[-0.2, 0.3], ylim=[-0.2, 0.3])

ax2.errorbar([0, 1], [method_a_avg.mean(), method_b_avg.mean()],
             [method_a_avg.sem(), method_b_avg.sem()], lw=2)
_, p = ttest_rel(method_a_avg, method_b_avg)
ax2.set(xticks=[0, 1], xticklabels=['Exp smoothing model', 'Optimal model'],
        ylabel='Decoding performance (r over chance)', title='p = %.3f' % p)

plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_optimal_vs_exp-fitted_prior'))
