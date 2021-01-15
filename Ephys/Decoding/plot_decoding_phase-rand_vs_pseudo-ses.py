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
from my_functions import paths, figure_style, get_full_region_name, get_parent_region_name

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
phase_rand = pd.read_pickle(join(SAVE_PATH,
       ('decode_%s_%s_phase-rand_%s_neurons_%s_sessions.p' % (TARGET, DECODER,
                                                              INCL_NEURONS, INCL_SESSIONS))))
pseudo_ses = pd.read_pickle(join(SAVE_PATH,
       ('decode_%s_%s_pseudo-blocks_%s_neurons_%s_sessions.p' % (TARGET, DECODER,
                                                              INCL_NEURONS, INCL_SESSIONS))))

# Exclude root
phase_rand = phase_rand.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(phase_rand['region']) if not j.islower()]
phase_rand = phase_rand.loc[incl_regions]
pseudo_ses = pseudo_ses.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(pseudo_ses['region']) if not j.islower()]
pseudo_ses = pseudo_ses.loc[incl_regions]

# Drop duplicates
pseudo_ses = pseudo_ses[pseudo_ses.duplicated(subset=['region', 'eid', 'probe'])
                                  == False]
phase_rand = phase_rand[phase_rand.duplicated(subset=['region', 'eid', 'probe'])
                                  == False]

# Get decoding performance over chance
pseudo_ses['acc_over_chance'] = (pseudo_ses['accuracy']
                                 - pseudo_ses['chance_accuracy']) * 100
phase_rand['acc_over_chance'] = (phase_rand['accuracy']
                                 - phase_rand['chance_accuracy']) * 100

# Get decoding average per region
pseudo_ses_avg = pseudo_ses.groupby('region').mean()['acc_over_chance']
phase_rand_avg = phase_rand.groupby('region').mean()['acc_over_chance']
pseudo_ses_p = pseudo_ses.groupby('region').mean()['p_accuracy']
phase_rand_p = phase_rand.groupby('region').mean()['p_accuracy']

# Get regions that are in both
pseudo_ses_avg = pseudo_ses_avg[pseudo_ses_avg.index.isin(phase_rand_avg.index)]
phase_rand_avg = phase_rand_avg[phase_rand_avg.index.isin(pseudo_ses_avg.index)]
pseudo_ses_p = pseudo_ses_p[pseudo_ses_p.index.isin(phase_rand_p.index)]
phase_rand_p = phase_rand_p[phase_rand_p.index.isin(pseudo_ses_p.index)]

# %%
figure_style(font_scale=1.8)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 8), dpi=150)

ax1.hist(pseudo_ses['chance_accuracy'], histtype='step', color=sns.color_palette('colorblind')[0],
         label='Pseudo blocks', lw=3)
ax1.hist(phase_rand['chance_accuracy'], histtype='step', color=sns.color_palette('colorblind')[1],
         label='Phase randomize', lw=3)
ax1.legend(frameon=False)
ax1.set(ylabel='Recordings per brain region count', xlabel='Decoding accuracy (%)',
        xlim=[0.35, 0.65], ylim=[0, 500], title='Comparision of chance level estimation')

for i in pseudo_ses.index:
    this_phase = phase_rand[(phase_rand['eid'] == pseudo_ses.loc[i, 'eid'])
                            & (phase_rand['region'] == pseudo_ses.loc[i, 'region'])
                            & (phase_rand['probe'] == pseudo_ses.loc[i, 'probe'])]
    if this_phase.shape[0] == 1:
        ax2.plot(pseudo_ses.loc[i, 'acc_over_chance'], this_phase['acc_over_chance'],
                 'o', color='b')
ax2.plot([-20, 20], [-20, 20], color='k')
ax2.set(xlim=[-20, 20], ylim=[-20, 20], xlabel='Decoding improvement over phase randomization (%)',
        ylabel='Decoding improvement over pseudo blocks (%)')

for i in pseudo_ses.index:
    this_phase = phase_rand[(phase_rand['eid'] == pseudo_ses.loc[i, 'eid'])
                            & (phase_rand['region'] == pseudo_ses.loc[i, 'region'])
                            & (phase_rand['probe'] == pseudo_ses.loc[i, 'probe'])]
    if this_phase.shape[0] == 1:
        ax3.plot(pseudo_ses.loc[i, 'p_accuracy'], this_phase['p_accuracy'], 'o', color='b')
#ax3.scatter(phase_rand_p, pseudo_ses_p)
ax3.plot([0, 1], [0, 1], color='k')
ax3.set(xlabel='Phase randomization p-value', ylabel='Pseudo blocks p-value')

plt.tight_layout(pad=2)
sns.despine()
plt.savefig(join(FIG_PATH, 'decode_phase-rand_vs_pseudo-blocks'))
