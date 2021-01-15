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
TARGET = 'prior'
DECODER = 'linear-regression'
MIN_R = 0.08
YLIM = 0.5
MIN_REC = 2
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 15
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
CHANCE_LEVEL = 'pseudo'
VALIDATION = 'kfold-interleaved'
FULL_NAME = False
PARENT_REGIONS = False
SAVE_FIG = False

# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                    INCL_SESSIONS, INCL_NEURONS))))

# Get decoding performance over chance
decoding_result['r_over_chance_infer'] = (decoding_result['r_infer']
                                          - decoding_result['r_infer_null'])
decoding_result['r_over_chance_block'] = (decoding_result['r_block']
                                          - decoding_result['r_block_null'])

# Get full region names
if PARENT_REGIONS:
    decoding_result['full_region'] = get_parent_region_name(decoding_result['region'].values)
else:
    decoding_result['full_region'] = get_full_region_name(decoding_result['region'].values)

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'r_mean_infer'] = decoding_result.loc[
                            decoding_result['region'] == region, 'r_over_chance_infer'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'r_mean_block'] = decoding_result.loc[
                            decoding_result['region'] == region, 'r_over_chance_block'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)

# %%
figure_style(font_scale=2)
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 8))
decoding_plot = decoding_result[(decoding_result['r_mean_infer'] >= MIN_R)
                                & (decoding_result['n_rec'] >= MIN_REC)]
if FULL_NAME:
    sort_regions = decoding_plot.groupby('full_region').max().sort_values(
                            'r_mean_infer', ascending=False).reset_index()['full_region']
    sns.barplot(x='r_over_chance_infer', y='full_region', data=decoding_plot,
                order=sort_regions, ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2",
                ax=ax1)
    sns.swarmplot(x='r_over_chance_infer', y='full_region', data=decoding_plot,
                order=sort_regions, ax=ax1)
else:
    sort_regions = decoding_plot.groupby('region').mean().sort_values(
                            'r_mean_infer', ascending=False).reset_index()['region']
    sns.barplot(x='r_over_chance_infer', y='region', data=decoding_plot,
                order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding improvement over chance (r)', ylabel='',
        xlim=[0, YLIM])
ax1.set(title='Regression of inferred stimulus prior from pre-stim activity')


ax2.hist(decoding_result['r_mean_infer'])
ax2.set(ylabel='Recordings', xlabel='Decoding improvement over chance (r)')

ax3.hist(decoding_result['r_infer_null'])
ax3.set(ylabel='Recordings', xlabel='Chance level decoding (r)')

ax4.scatter(decoding_result['r_mean_infer'], decoding_result['r_mean_block'])
ax4.plot([-0.1, 0.5], [-0.1, 0.5], color='k', lw=2)
_, p = ttest_rel(decoding_result['r_mean_infer'], decoding_result['r_mean_block'])
ax4.set(ylabel='Inferred stimulus prior (r)', xlabel='Actual stimulus prior (r)', xlim=[-0.1, 0.5],
        ylim=[-0.1, 0.5], title='p = %.3f' % p)

plt.tight_layout(pad=2)
sns.despine(trim=False)

if SAVE_FIG:
    plt.savefig(join(FIG_PATH, DECODER, '%s_%s_%s_%s_%s_cells' % (
                    TARGET, CHANCE_LEVEL, VALIDATION, INCL_SESSIONS, INCL_NEURONS)))
