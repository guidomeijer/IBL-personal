#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:37:05 2020

@author: guido
"""

import numpy as np
from os.path import join
from my_functions import paths, figure_style, get_full_region_name
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Settings
TARGET = 'block'
DECODER = 'bayes-multinomial'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
CHANCE_LEVEL = 'shuffle'
VALIDATION = 'kfold-interleaved'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
# GLM_PATH = '/home/guido/Data/Ephys/berk_glm_fits/completefits_2020-10-26.p'
GLM_PATH = '/home/guido/Data/Ephys/berk_glm_fits/completefits_2020-11-09.p'


def drop_subzero_max(row):
    maxind = row.tag_0
    if row[maxind] < 0:
        row.tag_0 = 'None'
    return row


# Get GLM target names
if TARGET == 'block':
    YLIM = [-0.001, 0.002]
    XLIM = [-10, 20]
    glm_target = 'pLeft'
elif TARGET == 'stim-side':
    glm_target = 'stimonR'
    XLIM = [-10, 50]
    YLIM = [-0.002, 0.006]
elif TARGET == 'reward':
    YLIM = [-0.01, 0.06]
    XLIM = [-10, 70]
    glm_target = 'correct'
elif TARGET == 'choice':
    glm_target = 'wheel'
    XLIM = [-10, 50]
    YLIM = [-0.004, 0.016]

# Load in GLM fit data
glm_fits = pd.read_pickle(GLM_PATH)
pleft_df = glm_fits['rawpoints'][glm_fits['rawpoints']['covname'] == glm_target]
pleft = pleft_df.groupby('nolayer_name').agg({'value':'median'}).squeeze()

# Throw out cells which do not have any kernel explain their activity more than a mean-rate model
nosubzero = glm_fits['masterscores'].apply(drop_subzero_max, axis=1)
bad_inds = nosubzero[nosubzero.tag_0 == 'None']
nosub_raw = glm_fits['rawpoints'].drop(index=bad_inds['cell'])
nosub_raw = nosub_raw[nosub_raw['covname'] == glm_target]
pleft_nosub = nosub_raw.groupby('nolayer_name').agg({'value':'median'}).squeeze()

# Load in decoding data
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
decoding_result['full_region'] = get_full_region_name(decoding_result['region'].values)

# Get decoding mean per region
acc_over_chance = decoding_result.groupby('full_region').mean()['acc_over_chance']

# Get regions that are in both
pleft = pleft[pleft.index.isin(acc_over_chance.index)]
acc_over_chance = acc_over_chance[acc_over_chance.index.isin(pleft.index)]

# Take exponent
#pleft = np.log(np.exp(pleft))

# Create merged dataframe
merged_df = pd.concat((acc_over_chance, pleft), axis=1)
merged_df = merged_df.sort_values(by=['value', 'acc_over_chance'], ascending=(False, False))
merged_df['ratio'] = ((merged_df['acc_over_chance'] / merged_df['acc_over_chance'].max())
                      * (merged_df['value'] / merged_df['value'].max()))
merged_df = merged_df[merged_df['acc_over_chance'] > 0]

# %%

if TARGET == 'block':
    fig_title = 'Stimulus prior decoding compared to pLeft kernel'
elif TARGET == 'stim-side':
    fig_title = 'Stimulus side decoding compared to stimonR kernel'
elif TARGET == 'reward':
    fig_title = 'Reward vs ommission decoding compared to correct kernel'
elif TARGET == 'choice':
    fig_title = 'Left vs right choice decoding compared to wheel kernel'

figure_style(font_scale=1.5)
f, ax1 = plt.subplots(1, 1, figsize=(6, 5), dpi=150, sharey=True)
ax1.scatter(acc_over_chance, pleft)
r, p = pearsonr(acc_over_chance, pleft)
m, b = np.polyfit(acc_over_chance, pleft, 1)
if p < 0.05:
    ax1.plot(np.arange(-50, 100, 0.1), m*np.arange(-50, 100, 0.1) + b, color='r')
ax1.set(xlabel='Decoding improvement over chance (% correct)',
        ylabel='Delta D squared of GLM fit',
        title='%s\n(r=%.2f, p=%.2f)' % (fig_title, r, p),
        xlim=XLIM, ylim=YLIM)

plt.tight_layout()
sns.despine(trim=True)
plt.savefig(join(FIG_PATH, 'Decoding', DECODER, 'decoding_vs_glm_%s' % TARGET))
