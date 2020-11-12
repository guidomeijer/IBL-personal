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
from ephys_functions import paths, figure_style

# Settings
TARGET = 'block'
DECODER = 'bayes'
MIN_PERF = 3
YLIM = 15
MIN_REC = 2
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'behavior_crit'  # all or aligned

# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH,
       ('decode_%s_%s_%s_neurons_%s_sessions.p' % (TARGET, DECODER, INCL_NEURONS, INCL_SESSIONS))))

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

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'acc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'acc_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)

# %%
figure_style(font_scale=2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), dpi=150)
decoding_plot = decoding_result[(decoding_result['acc_mean'] >= MIN_PERF)
                                & (decoding_result['n_rec'] >= MIN_REC)]
sort_regions = decoding_plot.groupby('region').mean().sort_values(
                                            'acc_mean', ascending=False).reset_index()['region']
sns.barplot(x='acc_over_chance', y='region', data=decoding_plot,
            order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding improvement over chance (% correct)', ylabel='',
        xlim=[0, YLIM])
if TARGET == 'stim_side':
    ax1.set(title='Decoding of stimulus side')
elif TARGET == 'block':
    ax1.set(title='Decoding of stimulus prior from pre-stim activity')
elif TARGET == 'blank':
    ax1.set(title='Decoding of stimulus prior from blank trials')
elif TARGET == 'block_stim':
    ax1.set(title='Decoding of stimulus prior from stimulus period')
elif TARGET == 'reward':
    ax1.set(title='Decoding of reward or ommission')
elif TARGET == 'choice':
    ax1.set(title='Decoding of choice')

ax2.hist(decoding_result['acc_over_chance'])
ax2.set(ylabel='Recordings', xlabel='Decoding improvement over chance (% correct)')

plt.tight_layout()
sns.despine(trim=False)

plt.savefig(join(FIG_PATH, 'decode_%s_%s_%s_neurons_%s_sessions_acc' % (
                                                TARGET, DECODER, INCL_NEURONS, INCL_SESSIONS)))
