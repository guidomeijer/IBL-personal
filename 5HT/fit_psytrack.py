# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import numpy as np
from psytrack.hyperOpt import hyperOpt
from psytrack.plot.analysisFunctions import makeWeightPlot
from oneibl.one import ONE
import matplotlib.pyplot as plt
import seaborn as sns


def fit_model(nickname, date_range, previous_trials=0):

    # Find sessions in ONE
    one = ONE()
    d_types = ['_iblrig_taskSettings.raw',
               'trials.probabilityLeft',
               'trials.contrastLeft',
               'trials.contrastRight',
               'trials.choice',
               'trials.feedbackType']
    eids = one.search(subject=nickname, date_range=[date_range[0], date_range[1]])
    if not eids:
        raise Exception('Session not found')

    # Initialize arrays
    contrast_l = np.zeros(0)
    contrast_r = np.zeros(0)
    choice = np.zeros(0)
    prob_l = np.zeros(0)
    n_trials = np.zeros(0)
    correct = np.zeros(0)

    # Load in data
    print('Loading session data from ONE..')
    for i, eid in enumerate(eids):
        d, this_prob_l, c_l, c_r, this_choice, feedback = one.load(
                    eid, d_types, dclass_output=False)

        # Exclude ommisions
        c_l = c_l[this_choice != 0]
        c_r = c_r[this_choice != 0]
        this_prob_l = this_prob_l[this_choice != 0]
        feedback = feedback[this_choice != 0]
        this_choice = this_choice[this_choice != 0]

        # Append to arrays
        contrast_l = np.append(contrast_l, c_l)
        contrast_r = np.append(contrast_r, c_r)
        choice = np.append(choice, this_choice)
        prob_l = np.append(prob_l, this_prob_l)
        correct = np.append(correct, feedback)
        n_trials = np.append(n_trials, np.size(this_choice))

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    contrast_l_log = contrast_l*100
    contrast_l_log[contrast_l_log == 0] = 0.1
    contrast_l_log = np.log10(contrast_l_log)
    contrast_r_log = contrast_r*100
    contrast_r_log[contrast_r_log == 0] = 0.1
    contrast_r_log = np.log10(contrast_r_log)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1 = contrast_l
    s2 = contrast_r
    for i in range(1, 10):
        s1 = np.column_stack((s1, np.append([contrast_l[0]]*(i+i), contrast_l[i:-i])))
        s2 = np.column_stack((s2, np.append([contrast_r[0]]*(i+i), contrast_r[i:-i])))
    s1_log = contrast_l_log
    s2_log = contrast_r_log
    for i in range(1, 10):
        s1_log = np.column_stack((s1_log, np.append([contrast_l_log[0]]*(i+i),
                                                    contrast_l_log[i:-i])))
        s2_log = np.column_stack((s2_log, np.append([contrast_r_log[0]]*(i+i),
                                                    contrast_r_log[i:-i])))

    # Create input dict
    D = {'name': nickname,
         'y': choice,
         'correct': correct,
         'dayLength': n_trials,
         'inputs': {'s1': s1_log, 's2': s2_log}
         }

    # Model parameters
    weights = {'bias': 1,
               's1': previous_trials+1,
               's2': previous_trials+1}
    K = np.sum([weights[i] for i in weights.keys()])
    if np.size(n_trials) == 1:
        hyper = {'sigInit': 2**4.,
                 'sigma': [2**-4.]*K,
                 'sigDay': None}
    else:
        hyper = {'sigInit': 2**4.,
                 'sigma': [2**-4.]*K,
                 'sigDay': [2**-4.]*K}
    optList = ['sigma']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)
    return wMode, prob_l


def plot_psytrack(wMode, prob_l):

    f, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    block_switch = np.where(np.abs(np.diff(prob_l)) > 0.1)[0]
    block_switch = np.concatenate(([0], block_switch+1, [np.size(prob_l)]), axis=0)
    for i, ind in enumerate(block_switch[:-1]):
        if prob_l[block_switch[i]] == 0.5:
            ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                     [-4, 4, 4, -4], color=[0.7, 0.7, 0.7])
        if prob_l[block_switch[i]] == 0.2:
            ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                     [-4, 4, 4, -4], color=[0.6, 0.6, 1])
        if prob_l[block_switch[i]] == 0.8:
            ax1.fill([block_switch[i], block_switch[i], block_switch[i+1], block_switch[i+1]],
                     [-4, 4, 4, -4], color=[1, 0.6, 0.6])
    ax1.plot(wMode[0], color='k', lw=3)
    ax1.plot(wMode[1], color='r', lw=3)
    ax1.plot(wMode[2], color='b', lw=3)
    ax1.legend(['Bias', 'Left stimulus', 'Right stimulus'], fontsize=12)
    ax1.set(ylabel='Weight', xlabel='Trials')
    sns.set(context='paper', font_scale=1.5, style='ticks')
    sns.despine(trim=True)
    plt.tight_layout(pad=2)
