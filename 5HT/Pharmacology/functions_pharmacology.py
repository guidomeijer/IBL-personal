#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:22:07 2020

@author: guido
"""

from oneibl.one import ONE
import numpy as np
from psytrack.hyperOpt import hyperOpt
import matplotlib.pyplot as plt
import seaborn as sns


def paths():
    data_path = '/home/guido/Data/5HT'
    fig_path = '/home/guido/Figures/5HT/Pharmacology'
    save_path = '/home/guido/Data/5HT'
    return data_path, fig_path, save_path


def load_session_one(nickname, date):
    # Find session in ONE
    one = ONE()
    d_types = ['_iblrig_taskSettings.raw',
               'trials.probabilityLeft',
               'trials.contrastLeft',
               'trials.contrastRight',
               'trials.choice',
               'trials.feedbackType']
    eid = one.search(subject=nickname, date_range=date, task_protocol='biased')
    if len(eid) != 1:
        raise Exception('Error loading session')
    d, prob_l, contrast_l, contrast_r, choice, correct = one.load(eid[0], d_types,
                                                                  dclass_output=False)

    # Exclude ommisions
    contrast_l = contrast_l[choice != 0]
    contrast_r = contrast_r[choice != 0]
    prob_l = prob_l[choice != 0]
    correct = correct[choice != 0]
    choice = choice[choice != 0]

    return contrast_l, contrast_r, prob_l, correct, choice


def fit_psytrack(nickname, date, previous_trials=0):

    # Load data
    contrast_l, contrast_r, prob_l, correct, choice = load_session_one(nickname, date)

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    p = 3.5
    contrast_l_transform = np.tanh(contrast_l * p) / np.tanh(p)
    contrast_r_transform = np.tanh(contrast_r * p) / np.tanh(p)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1_trans = contrast_l_transform
    s2_trans = contrast_r_transform
    for i in range(1, 10):
        s1_trans = np.column_stack((s1_trans, np.append([contrast_l_transform[0]]*(i+i),
                                                        contrast_l_transform[i:-i])))
        s2_trans = np.column_stack((s2_trans, np.append([contrast_r_transform[0]]*(i+i),
                                                        contrast_r_transform[i:-i])))

    # Create input dict
    D = {'name': nickname,
         'y': choice,
         'correct': correct,
         'dayLength': choice.shape[0],
         'inputs': {'s1': s1_trans, 's2': s2_trans}
         }

    # Model parameters
    weights = {'bias': 1,
               's1': previous_trials+1,
               's2': previous_trials+1}
    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
             'sigma': [2**-4.]*K,
             'sigDay': None}
    optList = ['sigInit', 'sigma']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)

    return wMode, prob_l, hyp


def fit_psytrack_multiple_days(nickname, dates, previous_trials=0):

    # Load data
    for i, date in enumerate(dates):
        if i == 0:
            contrast_l, contrast_r, prob_l, correct, choice = load_session_one(nickname, date)
            day_length = contrast_l.shape[0]
        else:
            c_l, c_r, p_l, cr, ch = load_session_one(nickname, date)
            contrast_l = np.append(contrast_l, c_l)
            contrast_r = np.append(contrast_r, c_r)
            prob_l = np.append(contrast_l, p_l)
            correct = np.append(correct, cr)
            choice = np.append(choice, ch)
            day_length = np.append(day_length, contrast_l.shape[0])

    # Change values to what the model input
    choice[choice == 1] = 2
    choice[choice == -1] = 1
    correct[correct == -1] = 0
    contrast_l[np.isnan(contrast_l)] = 0
    contrast_r[np.isnan(contrast_r)] = 0

    # Transform visual contrast
    p = 3.5
    contrast_l_transform = np.tanh(contrast_l * p) / np.tanh(p)
    contrast_r_transform = np.tanh(contrast_r * p) / np.tanh(p)

    # Reformat the stimulus vectors to matrices which include previous trials
    s1_trans = contrast_l_transform
    s2_trans = contrast_r_transform
    for i in range(1, 10):
        s1_trans = np.column_stack((s1_trans, np.append([contrast_l_transform[0]]*(i+i),
                                                        contrast_l_transform[i:-i])))
        s2_trans = np.column_stack((s2_trans, np.append([contrast_r_transform[0]]*(i+i),
                                                        contrast_r_transform[i:-i])))

    # Create input dict
    D = {'name': nickname,
         'y': choice,
         'correct': correct,
         'dayLength': day_length,
         'inputs': {'s1': s1_trans, 's2': s2_trans}
         }

    # Model parameters
    weights = {'bias': 1,
               's1': previous_trials+1,
               's2': previous_trials+1}
    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
             'sigma': [2**-4.]*K,
             'sigDay': [2**-4.]*K}
    optList = ['sigInit', 'sigma', 'sigDay']

    # Fit model
    print('Fitting model..')
    hyp, evd, wMode, hess = hyperOpt(D, hyper, weights, optList)

    return wMode, prob_l, hyp


def plot_psytrack(wMode, prob_l, plot_stim=True):

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
    if plot_stim is True:
        ax1.plot(wMode[1], color='r', lw=3)
        ax1.plot(wMode[2], color='b', lw=3)
    ax1.legend(['Bias', 'Left stimulus', 'Right stimulus'], fontsize=12)
    ax1.set(ylabel='Weight', xlabel='Trials')
    sns.set(context='paper', font_scale=1.5, style='ticks')
    sns.despine(trim=True)
    plt.tight_layout(pad=2)
    return ax1
