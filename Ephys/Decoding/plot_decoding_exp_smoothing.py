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
TARGET = 'prior-stimside'
CHANCE_LEVEL = 'pseudo'
DECODER = 'linear-regression'
INCL_NEURONS = 'all'
INCL_SESSIONS = 'aligned-behavior'
VALIDATION = 'kfold'
ATLAS = 'allen-atlas'
SHOW_REGIONS = 25
#SHOW_REGIONS = 'significant'
MIN_REC = 5
YLIM = [0, .4]
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Ephys', 'Decoding')
FULL_NAME = True
PARENT_REGIONS = False
SAVE_FIG = True

# %% Plot
# Load in data
decoding_result = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS))))


# Get decoding performance over chance
decoding_result['r_over_chance_prior'] = (decoding_result['r_prior']
                                          - decoding_result['r_prior_null'])
decoding_result['r_over_chance_block'] = (decoding_result['r_block']
                                          - decoding_result['r_block_null'])

# Get full region names
if PARENT_REGIONS:
    decoding_result['full_region'] = get_parent_region_name(decoding_result['region'].values)
else:
    decoding_result['full_region'] = get_full_region_name(decoding_result['region'].values)


# Exclude ventricles
decoding_result = decoding_result.reset_index(drop=True)
decoding_result = decoding_result.drop(index=[i for i, j in enumerate(decoding_result['full_region']) if 'ventricle' in j])

# Calculate average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    decoding_result.loc[decoding_result['region'] == region, 'r_mean_prior'] = decoding_result.loc[
                            decoding_result['region'] == region, 'r_over_chance_prior'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'r_mean_block'] = decoding_result.loc[
                            decoding_result['region'] == region, 'r_over_chance_block'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'n_rec'] = np.sum(
                                                            decoding_result['region'] == region)
    _, p = wilcoxon(decoding_result.loc[decoding_result['region'] == region, 'r_over_chance_prior'])
    decoding_result.loc[decoding_result['region'] == region, 'p_value'] = p


# %% Plot

# Drop areas with too few recordings and sort
decoding_plot = decoding_result[decoding_result['n_rec'] >= MIN_REC].copy()
decoding_plot = decoding_plot.sort_values('r_mean_prior', ascending=False)

if SHOW_REGIONS == 'significant':
    decoding_plot = decoding_plot[decoding_plot['p_value'] < 0.05]
else:
    decoding_plot = decoding_plot[decoding_plot['r_mean_prior']
                                  >= decoding_plot['r_mean_prior'].unique()[SHOW_REGIONS - 1]]

figure_style(font_scale=1.6)
f = plt.figure(figsize=(30, 10), dpi=150)
gs = f.add_gridspec(2, 4)
ax1 = f.add_subplot(gs[:, 0])
if FULL_NAME:
    sort_regions = decoding_plot.groupby('full_region').max().sort_values(
                            'r_mean_prior', ascending=False).reset_index()['full_region']
    sns.barplot(x='r_over_chance_prior', y='full_region', data=decoding_plot,
                order=sort_regions, ci=68, facecolor=(1, 1, 1, 0), errcolor=".2", edgecolor=".2",
                ax=ax1)
    sns.stripplot(x='r_over_chance_prior', y='full_region', data=decoding_plot,
                order=sort_regions, s=6, ax=ax1)
else:
    sort_regions = decoding_plot.groupby('region').mean().sort_values(
                            'r_mean_prior', ascending=False).reset_index()['region']
    sns.barplot(x='r_over_chance_prior', y='region', data=decoding_plot,
                order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding improvement over chance (r)', ylabel='',
        xlim=YLIM)
ax1.set(title='Decoding of stimulus prior (exp. smoothing stimulus sides)')

ax2 = f.add_subplot(gs[0, 1])
ax2.hist(decoding_result['r_mean_prior'], bins=30)
ax2.set(ylabel='Recordings', xlabel='Decoding improvement over chance (r)')

ax3 = f.add_subplot(gs[0, 2])
ax3.hist(decoding_result['r_prior_null'], bins=30)
ax3.set(ylabel='Recordings', xlabel='Chance level decoding (r)')

ax4 = f.add_subplot(gs[1, 1])
tau = decoding_result.groupby('subject').mean()['tau']
tau = tau[tau < 30]
ax4.hist(tau, bins=30)
ax4.set(ylabel='Mice', xlabel='Length of integration window (tau)', xlim=[0, 30])

ax5 = f.add_subplot(gs[1, 2])
# sns.kdeplot(x='r_over_chance_prior', y='r_over_chance_block', data=decoding_result, fill=True)
# ax5.scatter(decoding_result['r_over_chance_prior'], decoding_result['r_over_chance_block'])
# ax5.scatter(decoding_result['r_mean_prior'], decoding_result['r_mean_block'])
sns.kdeplot(x='r_mean_block', y='r_mean_prior', data=decoding_result, fill=True)
ax5.plot([-0.1, 0.2], [-0.1, 0.2], color='k', lw=2)
_, p = ttest_rel(decoding_result['r_mean_prior'].unique(), decoding_result['r_mean_block'].unique())
ax5.set(ylabel='Prior decoding over chance (r)', xlabel='Block decoding over chance (r)', xlim=[-0.1, 0.2],
        ylim=[-0.1, 0.2], title='r pior=%.3f, r block=%.3f, p=%.2f' % (
            decoding_result['r_mean_prior'].unique().mean(), decoding_result['r_mean_block'].unique().mean(), p))


ax6 = f.add_subplot(gs[0, 3])
ax6.hist(decoding_result['n_neurons'], bins=100)
ax6.set(ylabel='Recordings', xlabel='Number of neurons', xlim=[0, 500])

ax7 = f.add_subplot(gs[1, 3])
# ax7.scatter(decoding_result['n_neurons'], decoding_result['r_over_chance_prior'])
sns.kdeplot(x='n_neurons', y='r_over_chance_prior', data=decoding_result, fill=True)
m, b = np.polyfit(decoding_result['n_neurons'], decoding_result['r_over_chance_prior'], 1)
ax7.plot(np.arange(decoding_result['n_neurons'].max()),
         m*np.arange(decoding_result['n_neurons'].max()) + b,
         lw=2, color='r')
r, p = pearsonr(decoding_result['n_neurons'], decoding_result['r_over_chance_prior'])
ax7.set(ylabel='Decoding improvement over chance (r)', xlabel='Number of neurons',
        title='r=%.3f, p=%.3f' % (r, p), xlim=[0, 400], ylim=[-0.5, 0.5])

plt.tight_layout(pad=4)
sns.despine(trim=False)

if SAVE_FIG:
    plt.savefig(join(FIG_PATH, DECODER, '%s_%s_%s_%s_%s_cells_%s' % (
                    TARGET, CHANCE_LEVEL, VALIDATION, INCL_SESSIONS, INCL_NEURONS, ATLAS)))