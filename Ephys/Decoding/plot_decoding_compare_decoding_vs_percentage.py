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
TITLE_STR = 'Comparison of decoding vs percentage of neurons'
SAVE_STR = 'perc_block_neurons_vs_decoding_actions'

# %% Plot
# Load in data
decoding = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'linear-regression',
                            'prior-prevaction_pseudo_kfold_aligned-behavior_all_cells_beryl-atlas.p'))
percentage = pd.read_csv(join(SAVE_PATH, 'Ephys', 'block_neurons.csv'))

# Get decoding performance over chance
decoding['r_over_chance'] = (decoding['r_prior'] - decoding['r_prior_null'])

# Get percentage per region
percentage_all = (percentage.groupby('region').sum()['n_sig_neurons']
                  / percentage.groupby('region').sum()['n_neurons']) * 100

# Get average per region
decoding_avg = decoding.groupby('region').mean()['r_prior']
percentage_avg = percentage.groupby('region').mean()['percentage']
decoding_avg = decoding_avg[decoding_avg.index.isin(percentage_avg.index)]
percentage_avg = percentage_avg[percentage_avg.index.isin(decoding_avg.index)]
percentage_all = percentage_all[percentage_all.index.isin(decoding_avg.index)]
decoding_over = decoding.groupby('region').mean()['r_over_chance']
decoding_over = decoding_over[decoding_over.index.isin(percentage_avg.index)]


# %%
figure_style(font_scale=1.7)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10), dpi=150)
f.suptitle(TITLE_STR, fontsize=21)

ax1.scatter(decoding_avg, percentage_avg)
m, b = np.polyfit(decoding_avg, percentage_avg, 1)
ax1.plot(np.arange(-0.2, 0.6, 0.01), m*np.arange(-0.2, 0.6, 0.01) + b, lw=2, color='k')
r, p = pearsonr(decoding_avg, percentage_avg)
#ax1.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
ax1.set(xlabel='Decoding performance (r)', ylabel='% significant neurons',
        title=f'Pearson, r={r:.2f}, p={p:.2f}')

ax2.scatter(decoding_over, percentage_avg)
m, b = np.polyfit(decoding_over, percentage_avg, 1)
ax2.plot(np.arange(-0.2, 0.6, 0.01), m*np.arange(-0.2, 0.6, 0.01) + b, lw=2, color='k')
r, p = pearsonr(decoding_over, percentage_avg)
#ax1.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
ax2.set(xlabel='Decoding improvement over chance (r)', ylabel='% significant neurons (r)',
        title='Pearson, r=%.2f, p=%.2f' % (r, p))


plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_%s' % SAVE_STR))
