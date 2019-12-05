# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:23 2019

@author: guido
"""

import numpy as np
from psytrack.hyperOpt import hyperOpt
from oneibl.one import ONE


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

    # Reformat the stimulus vectors to matrices which include previous trials
    s1 = contrast_l
    s2 = contrast_r
    for i in range(1, 10):
        s1 = np.column_stack((s1, np.append([contrast_l[0]]*(i+i), contrast_l[i:-i])))
        s2 = np.column_stack((s2, np.append([contrast_r[0]]*(i+i), contrast_r[i:-i])))

    # Create input dict
    D = {'name': nickname,
         'y': choice,
         'correct': correct,
         'dayLength': n_trials,
         'inputs': {'s1': s1, 's2': s2}
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
