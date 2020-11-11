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
MIN_PERF_ACC = 3
MIN_PERF_F1 = 0.05
MIN_REC = 2
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'all'  # all or aligned

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
decoding_result['acc_over_chance'] = (
    decoding_result['accuracy'] - [i.mean() for i in decoding_result['chance_accuracy']]) * 100
decoding_result['f1_over_chance'] = (
                decoding_result['f1'] - [i.mean() for i in decoding_result['chance_f1']])

# Get significant decoding
for i in decoding_result.index:
    decoding_result.loc[i, 'acc_p'] = (np.sum(decoding_result.loc[i, 'chance_accuracy']
                                             > decoding_result.loc[i,'accuracy'])
                                       / decoding_result.loc[i,'chance_accuracy'].shape[0])

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'acc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'acc_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'f1_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'f1_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'auroc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'auroc'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)

figure_style(font_scale=1.5)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), dpi=150)
decoding_plot = decoding_result[(decoding_result['acc_mean'] >= MIN_PERF_ACC)
                                & (decoding_result['n_rec'] >= MIN_REC)]
sort_regions = decoding_plot.groupby('region').mean().sort_values(
                                            'acc_mean', ascending=False).reset_index()['region']
sns.barplot(x='acc_over_chance', y='region', data=decoding_plot,
            order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding of stimulus prior\n(% correct over chance level)', ylabel='',
        xlim=[0, 15], title='%s neurons, %s sessions' % (INCL_NEURONS, INCL_SESSIONS))

ax2.hist(decoding_result['acc_over_chance'])
ax2.set(ylabel='Recording count', xlabel='Decoding accuracy over chance', title='% correct')

plt.savefig(join(FIG_PATH, 'decode_%s_%s_%s_neurons_%s_sessions_acc' % (
                                                TARGET, DECODER, INCL_NEURONS, INCL_SESSIONS)))

"""
figure_style(font_scale=1.5)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10), dpi=150)
decoding_plot = decoding_result[(decoding_result['f1_mean'] >= MIN_PERF_F1)
                                & (decoding_result['n_rec'] >= MIN_REC)]
sort_regions = decoding_plot.groupby('region').mean().sort_values(
                                            'f1_mean', ascending=False).reset_index()['region']
sns.barplot(x='f1_over_chance', y='region', data=decoding_plot,
            order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding accuracy of stimulus prior\n(f1-score over chance level)', ylabel='',
        xlim=[0, 0.3], title='%s neurons, %s sessions' % (INCL_NEURONS, INCL_SESSIONS))

ax2.hist(decoding_result['f1_over_chance'])
ax2.set(ylabel='Recording count', xlabel='Decoding accuracy over chance', title='f1-score')

plt.savefig(join(FIG_PATH, 'decode_block_%s_%s_neurons_%s_sessions_f1' % (
                                                DECODER, INCL_NEURONS, INCL_SESSIONS)))
"""
