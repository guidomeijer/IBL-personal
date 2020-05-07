#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:40:36 2020

@author: guido
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions_pharmacology import paths, fit_psytrack_multiple_days

# Load data
sessions = pd.read_csv('pharmacology_sessions.csv', header=1)
results = pd.DataFrame(columns=['Nickname', 'Condition', 'Bias'])
for i, nickname in enumerate(sessions['Nickname'].unique()):
    print('Starting subject %s' % nickname)
    for j, condition in enumerate(['Pre-vehicle', 'Drug', 'Post-vehicle']):

        # Fit psytrack model over all weeks
        print('Condition: %s' % condition)
        wMode, prob_l, hyp = fit_psytrack_multiple_days(
                                sessions.loc[i, 'Nickname'],
                                sessions.loc[sessions['Nickname'] == nickname, condition].values)
        results.loc[results.shape[0]+1] = ([nickname] + [condition] + [hyp['sigma'][0]])

f, ax1 = plt.subplots(1, 1, figsize=(6, 6))
sns.lineplot(x='Condition', y='Bias', data=results, hue='Nickname', units='Nickname',
             estimator=None, sort=False)
# sns.lineplot(x='Condition', y='Bias', data=results, ci=68)
