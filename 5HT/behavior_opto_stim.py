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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_functions import (load_opto_trials, plot_psychometric, fit_prob_choice_model, paths,
                          fit_psytrack)
from oneibl.one import ONE
one = ONE()

# Settings
PRE_TRIALS = 1
POST_TRIALS = 5
DOWNLOAD_TRIALS = False
_, fig_path, _ = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')

    # Get trials DataFrame
    trials = pd.DataFrame()
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            these_trials = load_opto_trials(eid, DOWNLOAD_TRIALS)
            these_trials['session'] = ses_count
            trials = trials.append(these_trials, ignore_index=True)
            ses_count = ses_count + 1
        except Exception:
            print('Could not load trials')

    # Fit probabilistic choice model
    weights_stim = fit_prob_choice_model(trials[trials['laser_stimulation'] == 1],
                                         previous_trials=1)
    weights_no_stim = fit_prob_choice_model(trials[trials['laser_stimulation'] == 0],
                                            previous_trials=1)

    # Fit psytrack model
    #psytrack_stim = fit_psytrack(trials[trials['laser_stimulation'] == 1])
    #psytrack_no_stim = fit_psytrack(trials[trials['laser_stimulation'] == 0])

    # Get bias at block change points
    block_trans = pd.DataFrame()
    transitions = np.array(np.where(np.diff(trials['probabilityLeft']) != 0)[0]) + 1
    for j, trans in enumerate(transitions):
        if ((trials.loc[trans, 'probabilityLeft'] == 0.5)
                | (trials.loc[trans - 1, 'probabilityLeft'] == 0.5)
                | (trans + POST_TRIALS * 8 > trials['probabilityLeft'].shape[0])):
            continue
        pre_choices = trials.loc[np.where(trials['signed_contrast'] == 0)[0][
                np.where(trials['signed_contrast'] == 0)[0] < trans][-PRE_TRIALS:], 'right_choice']
        post_choices = trials.loc[np.where(trials['signed_contrast'] == 0)[0][
               np.where(trials['signed_contrast'] == 0)[0] >= trans][:POST_TRIALS], 'right_choice']
        block_trans = block_trans.append(pd.DataFrame(data={
                        'right_choice': np.append(pre_choices, post_choices),
                        'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                        'change_to': trials.loc[trans, 'probabilityLeft'],
                        'laser': trials.loc[trans, 'laser_stimulation']}))

    # Get bias at opto-stimulation change points
    laser_trans = pd.DataFrame()
    transitions = np.array(np.where(np.diff(trials['laser_stimulation']) != 0)[0]) + 1
    for j, trans in enumerate(transitions):
        if ((trials.loc[trans, 'probabilityLeft'] == 0.5)
                | (trials.loc[trans - 1, 'probabilityLeft'] == 0.5)
                | (trans + POST_TRIALS * 8 > trials['probabilityLeft'].shape[0])):
            continue
        pre_choices = trials.loc[np.where(trials['signed_contrast'] == 0)[0][
                np.where(trials['signed_contrast'] == 0)[0] < trans][-PRE_TRIALS:], 'right_choice']
        post_choices = trials.loc[np.where(trials['signed_contrast'] == 0)[0][
               np.where(trials['signed_contrast'] == 0)[0] >= trans][:POST_TRIALS], 'right_choice']
        laser_trans = laser_trans.append(pd.DataFrame(data={
                        'right_choice': np.append(pre_choices, post_choices),
                        'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                        'prob_left': trials.loc[trans, 'probabilityLeft'],
                        'laser': trials.loc[trans, 'laser_stimulation']}))

    # Plot
    sns.set(context='talk', style='ticks', font_scale=1.5)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 30))

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

    sns.lineplot(x='trial', y='right_choice', data=block_trans, style='laser',
                 hue='change_to', ci=0, ax=ax3, palette='colorblind')
    ax3.set(xticks=np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
            xticklabels=np.append(np.arange(-PRE_TRIALS, 0), np.arange(1, POST_TRIALS + 1)),
            ylabel='Fraction of right choices', xlabel='0% trials relative to block switch')

    sns.lineplot(x='trial', y='right_choice', data=block_trans[block_trans['laser'] == 1],
                 hue='change_to', ci=68, ax=ax4,
                 palette='colorblind')
    ax4.set(xticks=np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
            xticklabels=np.append(np.arange(-PRE_TRIALS, 0), np.arange(1, POST_TRIALS + 1)),
            ylabel='Fraction of right choices', xlabel='0% trials relative to block switch')

    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_opto_behavior' % nickname))