#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 13:33:56 2021

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
import matplotlib.pyplot as plt
from behavior_models import utils
from my_functions import query_sessions, paths, figure_style
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action

# Save path
_, fig_path, save_path = paths()

# Query all sessions that are aligned and meet behavioral criterion
eids, probes, subjects = query_sessions(selection='aligned-behavior', return_subjects=True)

# Loop over subjects
results = pd.DataFrame()
for i, subject in enumerate(np.unique(subjects)):
    print('\nStarting subject %s [%d of %d]\n' % (subject, i + 1, len(np.unique(subjects))))
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    for j, eid in enumerate(eids[subjects == subject]):
        data = utils.load_session(eid)
        if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
            stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            session_uuids.append(eid)
    print('\nLoaded data from %d sessions' % (j + 1))

    # Get maximum number of trials across sessions
    max_len = np.array([len(stimuli_arr[k]) for k in range(len(stimuli_arr))]).max()

    # Pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials
    stimuli = np.array([np.concatenate((stimuli_arr[k], np.zeros(max_len-len(stimuli_arr[k]))))
                        for k in range(len(stimuli_arr))])
    actions = np.array([np.concatenate((actions_arr[k], np.zeros(max_len-len(actions_arr[k]))))
                        for k in range(len(actions_arr))])
    stim_side = np.array([np.concatenate((stim_sides_arr[k],
                                          np.zeros(max_len-len(stim_sides_arr[k]))))
                          for k in range(len(stim_sides_arr))])
    session_uuids = np.array(session_uuids)

    # Fit previous stimulus side model
    model = exp_stimside(join(save_path, 'Behavior', 'exp_smoothing_model_fits/'),
                         session_uuids, subject, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=False)
    params_stimside = model.get_parameters(parameter_type='posterior_mean')
    priors_stimside = model.compute_prior(actions, stimuli, stim_side)[0]

    # Fit previous action model
    model = exp_prev_action(join(save_path, 'Behavior', 'exp_smoothing_model_fits/'),
                            session_uuids, subject, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=False)
    params_prevaction = model.get_parameters(parameter_type='posterior_mean')
    priors_prevaction = model.compute_prior(actions, stimuli, stim_side)[0]

    # Plot prior
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    figure_style()
    if len(stimuli_arr) == 1:
        ax1.plot(priors_stimside, lw=2)
    else:
        ax1.plot(priors_stimside[0][:len(stimuli_arr[0])], lw=2)
    ax1.set(xlabel='Trials', ylabel='Prior',
            title='Stimulus side model (alpha: %.2f)' % params_stimside[0])
    if len(stimuli_arr) == 1:
        ax2.plot(priors_prevaction, lw=2)
    else:
        ax2.plot(priors_prevaction[0][:len(stimuli_arr[0])], lw=2)
    ax2.set(xlabel='Trials', ylabel='Prior',
            title='Actions model (alpha: %.2f)' % params_prevaction[0])
    plt.savefig(join(fig_path, 'Behavior', 'exp_smoothing_model', subject))
    plt.close(f)

    # Add to dataframe
    results.loc[len(results) + 1, 'subject'] = subject
    results.loc[len(results), 'stim-side_alpha'] = params_stimside[0]
    results.loc[len(results), 'prev-action_alpha'] = params_prevaction[0]
    results.to_csv(join(save_path, 'Behavior', 'exp_smoothing_model_results.csv'))
