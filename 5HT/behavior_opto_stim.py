#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import alf
import numpy as np
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from my_functions import load_opto_trials, plot_psychometric, fit_prob_choice_model, paths
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD_TRIALS = False
_, fig_path, _ = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')

    # Get trials DataFrame
    trials = pd.DataFrame()
    for j, eid in enumerate(eids):
        try:
            these_trials = load_opto_trials(eid, DOWNLOAD_TRIALS)
            trials = trials.append(these_trials, ignore_index=True)
        except Exception:
            print('Could not load trials')

    # Fit probabilistic choice model
    weights_stim = fit_prob_choice_model(trials[trials['laser_stimulation'] == 1],
                                         previous_trials=0)
    weights_no_stim = fit_prob_choice_model(trials[trials['laser_stimulation'] == 0],
                                            previous_trials=0)

    # Plot
    sns.set(context='talk', style='ticks', font_scale=1.6)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # plot_psychometric(trials[trials['probabilityLeft'] == 0.5], ax=axs[i], color='k')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['laser_stimulation'] == 0)], ax=ax1, color='b')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.8)
                             & (trials['laser_stimulation'] == 1)], ax=ax1,
                      color='b', linestyle='--')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                             & (trials['laser_stimulation'] == 0)], ax=ax1, color='r')
    plot_psychometric(trials[(trials['probabilityLeft'] == 0.2)
                             & (trials['laser_stimulation'] == 1)], ax=ax1,
                      color='r', linestyle='--')
    ax1.text(-25, 0.75, '20:80', color='r')
    ax1.text(25, 0.25, '80:20', color='b')
    ax1.set(title='dashed line = opto stim')


    ax2.plot([1, 2], [weights_no_stim['0.0625'], weights_stim['0.0625']], color='gray', zorder=-1)
    ax2.scatter([1], [weights_no_stim['0.0625']], c=['k'], zorder=1, label='No stim')
    ax2.scatter([2], [weights_stim['0.0625']], c=['deepskyblue'], zorder=1, label='Stim')
    ax2.plot([4, 5], [weights_no_stim['0.125'], weights_stim['0.125']], color='gray', zorder=-1)
    ax2.scatter([4, 5], [weights_no_stim['0.125'], weights_stim['0.125']], c=['k', 'deepskyblue'],
                zorder=1)
    ax2.plot([7, 8], [weights_no_stim['0.25'], weights_stim['0.25']], color='gray', zorder=-1)
    ax2.scatter([7, 8], [weights_no_stim['0.25'], weights_stim['0.25']], c=['k', 'deepskyblue'],
                zorder=1)
    ax2.plot([10, 11], [weights_no_stim['1'], weights_stim['1']], color='gray', zorder=-1)
    ax2.scatter([10, 11], [weights_no_stim['1'], weights_stim['1']], c=['k', 'deepskyblue'],
                zorder=1)
    ax2.plot([13, 14], [weights_no_stim['block'], weights_stim['block']], color='gray', zorder=-1)
    ax2.scatter([13, 14], [weights_no_stim['block'], weights_stim['block']],
                c=['k', 'deepskyblue'], zorder=1)
    ax2.set(xticks=[1.5, 4.5, 7.5, 10.5, 13.5], xticklabels=['6.25%', '12.5%', '25%', '100%',
                                                             'Block'],
            xlabel='Predictors', ylabel='Weight')
    ax2.legend(frameon=False)

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_opto_behavior' % nickname))