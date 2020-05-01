# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:40:23 2019

@author: guido
"""

from os.path import join
from scipy.io import loadmat
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from functions_pharmacology import paths

# Settings
DATA_PATH, FIG_PATH, _ = paths()

# Load in data
data = loadmat(join(DATA_PATH, 'guido_analysis_18mar2020.mat'))
X = data['X'][0]
parameters = data['pnames'][0]

results = pd.DataFrame(columns=['subject', 'condition', 'week', 'window_length', 'window_counts'])
for i in range(len(X)):
    for j in range(X[i].shape[1]):
        win_length = np.mean(X[i][0][j][:, parameters[i][0] == 'runlength-tau'])
        win_counts = np.mean(X[i][0][j][:, parameters[i][0] == 'beta-hyp'], axis=0)[1]
        results.loc[results.shape[0]+1] = ([i] + [np.mod(j, 3)]
                                           + [np.floor(j/3)+1] + [win_length] + [win_counts])

# Get bias normalized to pre-vehicle
results.loc[results['condition'] == 0, 'window_length_rel'] = (
                        results.loc[results['condition'] == 0, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)
results.loc[results['condition'] == 1, 'window_length_rel'] = (
                        results.loc[results['condition'] == 1, 'window_length'].values
                        / results.loc[results['condition'] == 0, 'window_length'].values)
results.loc[results['condition'] == 0, 'window_counts_rel'] = (
                        results.loc[results['condition'] == 0, 'window_counts'].values
                        / results.loc[results['condition'] == 0, 'window_counts'].values)
results.loc[results['condition'] == 1, 'window_counts_rel'] = (
                        results.loc[results['condition'] == 1, 'window_counts'].values
                        / results.loc[results['condition'] == 0, 'window_counts'].values)


# %% Plot results
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
sns.set(context='paper', font_scale=1.5, style='ticks')

ax1.plot([results.loc[results['condition'] == 0, 'window_length'],
          results.loc[results['condition'] == 1, 'window_length']], color=[0.6, 0.6, 0.6])
ax1.errorbar([0, 1],
             [results.loc[results['condition'] == 0, 'window_length'].mean(),
              results.loc[results['condition'] == 1, 'window_length'].mean()],
             yerr=[results.loc[results['condition'] == 0, 'window_length'].sem(),
                   results.loc[results['condition'] == 1, 'window_length'].sem()],
             color='black', lw=2)

ax2.plot([results.loc[results['condition'] == 0, 'window_counts'],
          results.loc[results['condition'] == 1, 'window_counts']], color=[0.6, 0.6, 0.6])
ax2.errorbar([0, 1],
             [results.loc[results['condition'] == 0, 'window_counts'].mean(),
              results.loc[results['condition'] == 1, 'window_counts'].mean()],
             yerr=[results.loc[results['condition'] == 0, 'window_counts'].sem(),
                   results.loc[results['condition'] == 1, 'window_counts'].sem()],
             color='black', lw=2)
