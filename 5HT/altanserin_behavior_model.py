# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fit_psytrack import fit_model

# Settings
TRIAL_WIN = [5, 10]

# Load in session dates
sessions = pd.read_csv('altanserin_sessions.csv', header=1, index_col=0)

# Load data
results = pd.DataFrame(columns=['subject', 'bias', 'first_bias', 'condition'])
for i, nickname in enumerate(sessions.index.values):

    # Fit model
    pre, prob_l_pre = fit_model(nickname, [sessions.loc[nickname, 'Pre-vehicle'],
                                           sessions.loc[nickname, 'Pre-vehicle']])
    drug, prob_l_drug = fit_model(nickname, [sessions.loc[nickname, 'Drug'],
                                             sessions.loc[nickname, 'Drug']])
    post, prob_l_post = fit_model(nickname, [sessions.loc[nickname, 'Post-vehicle'],
                                             sessions.loc[nickname, 'Post-vehicle']])

    # Create dataframes
    trial_vec = np.append(np.arange(-TRIAL_WIN[0], 0), np.arange(1, TRIAL_WIN[1]+1))
    pre_left = pd.DataFrame(columns=['bias', 'trial'])
    pre_right = pd.DataFrame(columns=['bias', 'trial'])
    for l, ind in enumerate(np.where(np.diff(prob_l_pre) > 0.5)[0]):
        pre_left = pre_left.append(
                pd.DataFrame({'bias': pre[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]],
                              'trial': trial_vec}))
    for l, ind in enumerate(np.where(np.diff(prob_l_pre) < -0.5)[0]):
        pre_right = pre_right.append(
                pd.DataFrame({'bias': pre[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]],
                              'trial': trial_vec}))
    drug_left = pd.DataFrame(columns=['bias', 'trial'])
    drug_right = pd.DataFrame(columns=['bias', 'trial'])
    for l, ind in enumerate(np.where(np.diff(prob_l_drug) > 0.5)[0]):
        drug_left = drug_left.append(
                pd.DataFrame({'bias': drug[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]],
                              'trial': trial_vec}))
    for l, ind in enumerate(np.where(np.diff(prob_l_drug) < -0.5)[0]):
        drug_right = drug_right.append(
                pd.DataFrame({'bias': drug[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]],
                              'trial': trial_vec}))
    post_left = pd.DataFrame(columns=['bias', 'trial'])
    post_right = pd.DataFrame(columns=['bias', 'trial'])
    for l, ind in enumerate(np.where(np.diff(prob_l_post) > 0.5)[0]):
        post_left = post_left.append(
                pd.DataFrame({'bias': post[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]],
                              'trial': trial_vec}))
    for l, ind in enumerate(np.where(np.diff(prob_l_post) < -0.5)[0]):
        post_right = post_right.append(
                pd.DataFrame({'bias': post[0][ind-TRIAL_WIN[0]:ind+TRIAL_WIN[1]],
                              'trial': trial_vec}))

    # Plot results
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    sns.lineplot(x='trial', y='bias', data=pre_left, ax=ax1)
    sns.lineplot(x='trial', y='bias', data=drug_left, ax=ax1)
    sns.lineplot(x='trial', y='bias', data=post_left, ax=ax1)
    plt.legend(['pre', 'drug', 'post'])

    sns.lineplot(x='trial', y='bias', data=pre_right, ax=ax2)
    sns.lineplot(x='trial', y='bias', data=drug_right, ax=ax2)
    sns.lineplot(x='trial', y='bias', data=post_right, ax=ax2)
    plt.legend(['pre', 'drug', 'post'])
