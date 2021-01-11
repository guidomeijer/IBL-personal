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
from ephys_functions import paths, figure_style, get_full_region_name, get_parent_region_name

# Settings
TARGET = 'block'
DECODER = 'bayes'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
FULL_NAME = True
PARENT_REGIONS = False

# %% Plot
# Load in data
block = pd.read_pickle(join(SAVE_PATH, 'bayes-multinomial',
                            'block_pseudo-session_kfold-interleaved_aligned-behavior_15_cells.p'))
prior = pd.read_pickle(join(SAVE_PATH, 'linear-regression',
                            'prior_pseudo_kfold-interleaved_aligned-behavior_15_cells.p'))

# Get decoding performance over chance
block['perf_over_chance'] = (block['accuracy'] - block['chance_accuracy']) * 100
prior['perf_over_chance_infer'] = (prior['r_infer'] - prior['r_infer_null'])
prior['perf_over_chance_block'] = (prior['r_block'] - prior['r_block_null'])

# Get decoding average per region
block_avg = block.groupby('region').mean()['perf_over_chance']
prior_infer_avg = prior.groupby('region').mean()['perf_over_chance_infer']
prior_block_avg = prior.groupby('region').mean()['perf_over_chance_block']

# Get regions that are in both
block_avg = block_avg[block_avg.index.isin(prior_infer_avg.index)]
prior_infer_avg = prior_infer_avg[prior_infer_avg.index.isin(block_avg.index)]
prior_block_avg = prior_block_avg[prior_block_avg.index.isin(block_avg.index)]

# %%
figure_style(font_scale=1.8)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), dpi=150)

for i in block.index:
    this_phase = prior[(prior['eid'] == block.loc[i, 'eid'])
                       & (prior['region'] == block.loc[i, 'region'])
                       & (prior['probe'] == block.loc[i, 'probe'])]
    if this_phase.shape[0] == 1:
        ax1.plot(block.loc[i, 'perf_over_chance'], this_phase['perf_over_chance_block'],
                 'o', color='b')
        ax2.plot(block.loc[i, 'perf_over_chance'], this_phase['perf_over_chance_infer'],
                 'o', color='b')
ax1.set(xlim=[-10, 10], ylim=[-0.5, 0.5], xlabel='Regression of actual prior (r over chance)',
        ylabel='Decoding of block identity (% correct over chance)')
ax2.set(xlim=[-10, 10], ylim=[-0.5, 0.5], xlabel='Regression of inferred prior (r over chance)',
        ylabel='Decoding of block identity (% correct over chance)')

plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_inferred_vs_actual_prior'))
