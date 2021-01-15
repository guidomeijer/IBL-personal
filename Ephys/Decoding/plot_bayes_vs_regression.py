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
from my_functions import paths, figure_style

# Settings
TARGET = 'block'
DECODER = 'regression'
MIN_PERF = 2
YLIM = 15
MIN_REC = 3
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'behavior_crit'  # all or aligned

# %% Plot
# Load in data
decoding_bayes = pd.read_pickle(join(SAVE_PATH,
       ('decode_%s_%s_%s_neurons_%s_sessions.p' % (TARGET, 'bayes', INCL_NEURONS, INCL_SESSIONS))))
decoding_regression = pd.read_pickle(join(SAVE_PATH,
        ('decode_%s_%s_%s_neurons_%s_sessions.p' % (TARGET, 'regression',
                                                    INCL_NEURONS, INCL_SESSIONS))))

# Exclude root
decoding_bayes = decoding_bayes.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_bayes['region']) if not j.islower()]
decoding_bayes = decoding_bayes.loc[incl_regions]
decoding_regression = decoding_regression.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_regression['region']) if not j.islower()]
decoding_regression = decoding_regression.loc[incl_regions]

# Drop duplicates
decoding_bayes = decoding_bayes[decoding_bayes.duplicated(subset=['region', 'eid', 'probe'])
                                == False]
decoding_regression = decoding_regression[decoding_regression.duplicated(
                                            subset=['region', 'eid', 'probe']) == False]

# Get decoding performance over chance
decoding_bayes['acc_over_chance'] = (decoding_bayes['accuracy']
                                     - decoding_bayes['chance_accuracy']) * 100
decoding_regression['acc_over_chance'] = (decoding_regression['accuracy']
                                          - decoding_regression['chance_accuracy']) * 100

# Get mean
bayes_over_chance = decoding_bayes.groupby('region').mean()['acc_over_chance']
reg_over_chance = decoding_regression.groupby('region').mean()['acc_over_chance']
bayes_mean = decoding_bayes.groupby('region').mean()['accuracy'] * 100
reg_mean = decoding_regression.groupby('region').mean()['accuracy'] * 100

# Get regions that are in both
bayes_over_chance = bayes_over_chance[bayes_over_chance.index.isin(reg_over_chance.index)]
reg_over_chance = reg_over_chance[reg_over_chance.index.isin(bayes_over_chance.index)]
bayes_mean = bayes_mean[bayes_mean.index.isin(reg_mean.index)]
reg_mean = reg_mean[reg_mean.index.isin(bayes_mean.index)]

# %%
figure_style(font_scale=1.2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
ax1.scatter(bayes_over_chance, reg_over_chance)
ax1.set(xlabel='Naive Bayes performance over chance (% correct)',
        ylabel='Logistic Regression over chance (% correct)',
        xlim=[-20, 20], ylim=[-20, 20])
ax1.plot([-20, 20], [-20, 20])


ax2.scatter(bayes_mean, reg_mean)
ax2.set(xlabel='Naive Bayes performance over chance (% correct)',
        ylabel='Logistic Regression over chance (% correct)',
        xlim=[30, 70], ylim=[30, 70])
ax2.plot([30, 70], [30, 70])

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'bayes_vs_regression'))

