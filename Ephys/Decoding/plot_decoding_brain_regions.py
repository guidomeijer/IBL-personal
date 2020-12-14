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
TARGET = 'stim-side'
DECODER = 'bayes-multinomial'
MIN_PERF = 20
YLIM = 50
MIN_REC = 2
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
CHANCE_LEVEL = 'pseudo-sessions'
VALIDATION = 'kfold-interleaved'
FULL_NAME = True
PARENT_REGIONS = False
SAVE_FIG = True

# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                    INCL_SESSIONS, INCL_NEURONS))))

# Exclude root
decoding_result = decoding_result.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Drop duplicates
decoding_result = decoding_result[decoding_result.duplicated(subset=['region', 'eid', 'probe'])
                                  == False]

# Get decoding performance over chance
decoding_result['acc_over_chance'] = (decoding_result['accuracy']
                                      - decoding_result['chance_accuracy']) * 100

# Get full region names
if PARENT_REGIONS:
    decoding_result['full_region'] = get_parent_region_name(decoding_result['region'].values)
else:
    decoding_result['full_region'] = get_full_region_name(decoding_result['region'].values)

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'acc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'acc_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)

# %%
figure_style(font_scale=1.4)
f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))
decoding_plot = decoding_result[(decoding_result['acc_mean'] >= MIN_PERF)
                                & (decoding_result['n_rec'] >= MIN_REC)]
if FULL_NAME:
    sort_regions = decoding_plot.groupby('full_region').max().sort_values(
                            'acc_mean', ascending=False).reset_index()['full_region']
    sns.barplot(x='acc_over_chance', y='full_region', data=decoding_plot,
                order=sort_regions, ci=68, ax=ax1)
else:
    sort_regions = decoding_plot.groupby('region').mean().sort_values(
                            'acc_mean', ascending=False).reset_index()['region']
    sns.barplot(x='acc_over_chance', y='region', data=decoding_plot,
                order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding improvement over chance (% correct)', ylabel='',
        xlim=[0, YLIM])
if TARGET == 'stim-side':
    ax1.set(title='Decoding of stimulus side on the left or right')
elif TARGET == 'block':
    ax1.set(title='Decoding of stimulus prior from pre-stim activity')
elif TARGET == 'blank':
    ax1.set(title='Decoding of stimulus prior from blank trials')
elif TARGET == 'block-stim':
    ax1.set(title='Decoding of stimulus prior from stimulus period')
elif TARGET == 'reward':
    ax1.set(title='Decoding of reward or ommission')
elif TARGET == 'choice':
    ax1.set(title='Decoding of motor response to left or right')

ax2.hist(decoding_result['acc_over_chance'])
ax2.set(ylabel='Recordings', xlabel='Decoding improvement over chance (% correct)')

ax3.hist(decoding_result['chance_accuracy'])
ax3.set(ylabel='Recordings', xlabel='Chance level decoding (% correct)')

ax4.hist(decoding_result['p_accuracy'])
ax4.set(ylabel='Recordings', xlabel='Decoding significance (p-value)')

plt.tight_layout(pad=2)
sns.despine(trim=False)

if SAVE_FIG:
    plt.savefig(join(FIG_PATH, DECODER, '%s_%s_%s_%s_%s_cells' % (
                    TARGET, CHANCE_LEVEL, VALIDATION, INCL_SESSIONS, INCL_NEURONS)))
