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
from scipy.stats import ttest_rel, pearsonr, wilcoxon
from my_functions import paths, figure_style, get_full_region_name, get_parent_region_name

# Settings
TARGET = 'prior-prevaction'
CHANCE_LEVEL = 'other-trials'
DECODER = 'linear-regression'
INCL_NEURONS = 'all'
INCL_SESSIONS = 'aligned-behavior'
VALIDATION = 'kfold'
ATLAS = 'beryl-atlas'
SHOW_REGIONS = 20
#SHOW_REGIONS = 'significant'
MIN_REC = 5
MIN_TOTAL_NEURONS = 0
MAX_TAU = 30
YLIM = [-.3, .51]
DPI = 150
TIME_WIN = '200-0'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Ephys', 'Decoding')
FULL_NAME = True
PARENT_REGIONS = False
SAVE_FIG = True
BLOCK = True
OVER_CHANCE = True

# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                                      f'{TARGET}_{CHANCE_LEVEL}_{VALIDATION}_{INCL_SESSIONS}_' \
                                      f'{INCL_NEURONS}_cells_{ATLAS}_{TIME_WIN}.p'))

# Get decoding performance over chance
if OVER_CHANCE:
    decoding_result['r_prior_plot'] = decoding_result['r_prior'] - decoding_result['r_prior_null']
    if BLOCK:
        decoding_result['r_block_plot'] = (decoding_result['r_block']
                                           - decoding_result['r_block_null'])
    else:
        decoding_result['r_block_plot'] = decoding_result['r_prior_plot']
else:
    decoding_result['r_prior_plot'] = decoding_result['r_prior']
    if BLOCK:
        decoding_result['r_block_plot'] = decoding_result['r_block']
    else:
        decoding_result['r_block_plot'] = decoding_result['r_prior']


# Get full region names
if PARENT_REGIONS:
    decoding_result['full_region'] = get_parent_region_name(decoding_result['region'].values)
else:
    decoding_result['full_region'] = get_full_region_name(decoding_result['region'].values)

# Exclude ventricles
decoding_result = decoding_result.reset_index(drop=True)
decoding_result = decoding_result.drop(index=[i for i, j in enumerate(decoding_result['full_region']) if 'ventricle' in j])

# Exclude mice with bad fit
decoding_result = decoding_result[decoding_result['tau'] <= MAX_TAU]

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'r_mean_prior'] = decoding_result.loc[
                            decoding_result['region'] == region, 'r_prior_plot'].median()
    decoding_result.loc[decoding_result['region'] == region, 'r_mean_block'] = decoding_result.loc[
                            decoding_result['region'] == region, 'r_block_plot'].median()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)
    decoding_result.loc[decoding_result['region'] == region, 'n_total_neurons'] = decoding_result.loc[
                            decoding_result['region'] == region, 'n_neurons'].sum()
    _, p = wilcoxon(decoding_result.loc[decoding_result['region'] == region, 'r_prior_plot'])
    decoding_result.loc[decoding_result['region'] == region, 'p_value'] = p

# Print some summaries
print('Number of recording sessions: %d\nNumber of recordings per brain region: %d\n'
      'Total number of neurons: %d' % (decoding_result.groupby('eid').sum().shape[0],
                                       decoding_result.shape[0],
                                       decoding_result['n_neurons'].sum()))

# %% Plot

# Drop areas with too few recordings and sort
decoding_plot = decoding_result[((decoding_result['n_rec'] >= MIN_REC)
                                 & (decoding_result['n_total_neurons'] >= MIN_TOTAL_NEURONS))].copy()
decoding_plot = decoding_plot.sort_values('r_mean_prior', ascending=False)

if SHOW_REGIONS == 'significant':
    decoding_plot = decoding_plot[decoding_plot['p_value'] < 0.05]
else:
    decoding_plot = decoding_plot[decoding_plot['r_mean_prior']
                                  >= decoding_plot['r_mean_prior'].unique()[SHOW_REGIONS - 1]]

figure_style(font_scale=2)
f = plt.figure(figsize=(30, 20), dpi=DPI)
gs = f.add_gridspec(3, 4)
if 'prior-prevaction' in TARGET:
    target_str = 'previous actions'
elif 'prior-stimsides' in TARGET:
    target_str = 'stimulus sides'
if VALIDATION == 'kfold':
    val_str = 'continuous 5-fold'
elif VALIDATION == 'kfold-interleaved':
    val_str = 'interleaved 5-fold'
f.suptitle('Decoding of %s using linear regression with %s cross-validation' % (
    target_str, val_str), fontsize=25)

ax1 = f.add_subplot(gs[:, 0])
if FULL_NAME:
    sort_regions = decoding_plot.groupby('full_region').max().sort_values(
                            'r_mean_prior', ascending=False).reset_index()['full_region']
    """
    sns.barplot(x='r_over_chance_prior', y='full_region', data=decoding_plot,
                order=sort_regions, ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2",
                ax=ax1)
    """

    ax_lines = sns.pointplot(x='r_prior_plot', y='full_region', data=decoding_plot,
                             order=sort_regions, ci=0, join=False, estimator=np.median, color='k',
                             markers="|", scale=2, ax=ax1)
    #plt.setp(ax_lines.lines, zorder=100)
    plt.setp(ax_lines.collections, zorder=100, label="")
    sns.stripplot(x='r_prior_plot', y='full_region', data=decoding_plot,
                order=sort_regions, s=6, ax=ax1)
    #sns.pointplot(x='r_prior_plot', y='full_region', data=decoding_plot,
    #            order=sort_regions, ci=68, join=False, estimator=np.median, color='k', ax=ax1)
else:
    sort_regions = decoding_plot.groupby('region').mean().sort_values(
                            'r_mean_prior', ascending=False).reset_index()['region']
    sns.barplot(x='r_prior_plot', y='region', data=decoding_plot,
                order=sort_regions, ci=68, ax=ax1)

ax1.plot([0, 0], ax1.get_ylim(), color=[0.5, 0.5, 0.5], ls='--')
if OVER_CHANCE:
    str_xlabel = 'Decoding improvement\nover pseudo sessions (r)'
else:
    str_xlabel = 'Decoding performance (r)'
ax1.set(xlabel=str_xlabel, ylabel='', xlim=YLIM)

ax2 = f.add_subplot(gs[0, 1])
ax2.hist(decoding_result.groupby('region').mean()['r_prior'], bins=30)
ax2.set(ylabel='Recordings', xlabel='r', title='Decoding performance')

ax3 = f.add_subplot(gs[0, 2])
if not np.isnan(decoding_result['r_prior_null'][0]):
    ax3.hist(decoding_result['r_prior_null'], bins=30)
    ax3.set(ylabel='Recordings', xlabel='r', title='Decoding of pseudo-sessions')

ax4 = f.add_subplot(gs[0, 3])
ax4.hist(decoding_result['r_mean_prior'], bins=50)
ax4.set(ylabel='Recordings', xlabel='r', title='Decoding improvement over pseudo-sessions',
        xlim=[-0.2, 0.3])

ax5 = f.add_subplot(gs[1, 1])
ax5.hist(decoding_result.groupby('subject').mean()['tau'], bins=30)
ax5.set(ylabel='Mice', xlabel='tau', title='Length of integration window', xlim=[0, 30])

ax6 = f.add_subplot(gs[1, 2])
ax6.scatter(decoding_result['r_mean_block'], decoding_result['r_mean_prior'])
ax6.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2, ls='--')
_, p = ttest_rel(decoding_result['r_mean_prior'].unique(), decoding_result['r_mean_block'].unique())
ax6.set(ylabel='Decoding %s\n(r over chance)' % target_str, xlabel='Decoding actual prior (r over chance)',
        title='Averaged per region\n(paired t-test, p=%.2f)' % p,
        xlim=[-0.2, 0.4], ylim=[-0.2, 0.4])

"""
ax7 = f.add_subplot(gs[1, 3])
sns.kdeplot(x='r_over_chance_block', y='r_over_chance_prior', data=decoding_result, fill=True, ax=ax7)
#ax7.scatter(decoding_result['r_over_chance_block'], decoding_result['r_over_chance_prior'])
ax7.plot([-0.5, 0.5], [-0.5, 0.5], color='k', lw=2)
_, p = ttest_rel(decoding_result['r_over_chance_prior'], decoding_result['r_over_chance_block'])
ax7.set(ylabel='Decoding %s\n(r over chance)' % target_str, xlabel='Decoding actual prior (r over chance)',
        xlim=[-0.2, 0.4], ylim=[-0.2, 0.4], title='All recordings per region\n(paired t-test, p=%.3f)' % p)
"""

ax7 = f.add_subplot(gs[1, 3])
ax7.hist(decoding_result['n_neurons'], bins=100)
ax7.set(ylabel='Recordings', title='Number of neurons per region', xlim=[0, 300])

max_neurons = 150
ax8 = f.add_subplot(gs[2, 1])
ax8.scatter(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
            decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior_plot'])
m, b = np.polyfit(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
                  decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior_plot'], 1)
ax8.plot(np.arange(decoding_result['n_neurons'].max()),
         m*np.arange(decoding_result['n_neurons'].max()) + b,
         lw=2, color='k')
r, p = pearsonr(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
                decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior_plot'])
ax8.set(ylabel='Decoding %s (r)' % target_str, xlabel='Number of neurons',
        title='Decoding improvement over chance\n(Pearson, r=%.2f, p=%.2f)' % (r, p), xlim=[0, max_neurons],
        ylim=[-0.5, 1])

ax9 = f.add_subplot(gs[2, 2])
ax9.scatter(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
            decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior'])
m, b = np.polyfit(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
                  decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior'], 1)
ax9.plot(np.arange(decoding_result['n_neurons'].max()),
         m*np.arange(decoding_result['n_neurons'].max()) + b,
         lw=2, color='k')
r, p = pearsonr(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
                decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior'])
ax9.set(ylabel='Decoding %s (r)' % target_str, xlabel='Number of neurons',
        title='Decoding performance\n(Pearson, r=%.2f, p=%.2f)' % (r, p), xlim=[0, max_neurons],
        ylim=[-0.5, 1])

ax10 = f.add_subplot(gs[2, 3])
ax10.scatter(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
             decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior_train'])
m, b = np.polyfit(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
                  decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior_train'], 1)
ax10.plot(np.arange(decoding_result['n_neurons'].max()),
         m*np.arange(decoding_result['n_neurons'].max()) + b,
         lw=2, color='k')
r, p = pearsonr(decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'n_neurons'],
                decoding_result.loc[decoding_result['n_neurons'] < max_neurons, 'r_prior_train'])
ax10.set(ylabel='Decoding %s (r)' % target_str, xlabel='Number of neurons',
        title='Decoding perf. on training set\n(Pearson, r=%.2f, p=%.2f)' % (r, p), xlim=[0, max_neurons],
        ylim=[-0.5, 1])

plt.tight_layout(pad=4)
sns.despine(trim=True)

if SAVE_FIG:
    plt.savefig(join(FIG_PATH, DECODER, '%s_%s_%s_%s_%s_cells_%s_%s' % (
                    TARGET, CHANCE_LEVEL, VALIDATION, INCL_SESSIONS, INCL_NEURONS, ATLAS,
                    TIME_WIN)))