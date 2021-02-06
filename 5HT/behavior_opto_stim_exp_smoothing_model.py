#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:42:01 2021

@author: guido
"""

import numpy as np
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from my_functions import load_opto_trials, paths
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD_TRIALS = False
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')

    # Get trial data
    stimuli_arr_ns, actions_arr_ns, stim_sides_arr_ns, session_uuids = [], [], [], []
    stimuli_arr_s, actions_arr_s, stim_sides_arr_s = [], [], []
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            trials = load_opto_trials(eid, DOWNLOAD_TRIALS)
            stimuli_arr_ns.append(trials.loc[trials['laser_stimulation'] == 0,
                                             'signed_contrast'].values)
            actions_arr_ns.append(trials.loc[trials['laser_stimulation'] == 0, 'choice'].values)
            stim_sides_arr_ns.append(trials.loc[trials['laser_stimulation'] == 0,
                                                'stim_side'].values)
            stimuli_arr_s.append(trials.loc[trials['laser_stimulation'] == 1,
                                            'signed_contrast'].values)
            actions_arr_s.append(trials.loc[trials['laser_stimulation'] == 1, 'choice'].values)
            stim_sides_arr_s.append(trials.loc[trials['laser_stimulation'] == 1,
                                               'stim_side'].values)
            session_uuids.append(eid)
        except Exception:
            print('Could not load trials for %s' % eid)

    # get maximum number of trials across sessions
    max_len_ns = np.array([len(stimuli_arr_ns[k]) for k in range(len(stimuli_arr_ns))]).max()
    max_len_s = np.array([len(stimuli_arr_s[k]) for k in range(len(stimuli_arr_s))]).max()

    # pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials and convert to arrays
    stimuli_ns = np.array([np.concatenate((stimuli_arr_ns[k],
                                            np.zeros(max_len_ns-len(stimuli_arr_ns[k]))))
                           for k in range(len(stimuli_arr_ns))])
    actions_ns = np.array([np.concatenate((actions_arr_ns[k],
                                           np.zeros(max_len_ns-len(actions_arr_ns[k]))))
                           for k in range(len(actions_arr_ns))])
    stim_side_ns = np.array([np.concatenate((stim_sides_arr_ns[k],
                                             np.zeros(max_len_ns-len(stim_sides_arr_ns[k]))))
                             for k in range(len(stim_sides_arr_ns))])
    stimuli_s = np.array([np.concatenate((stimuli_arr_s[k],
                                          np.zeros(max_len_s-len(stimuli_arr_s[k]))))
                          for k in range(len(stimuli_arr_ns))])
    actions_s = np.array([np.concatenate((actions_arr_s[k],
                                          np.zeros(max_len_s-len(actions_arr_s[k]))))
                          for k in range(len(actions_arr_s))])
    stim_side_s = np.array([np.concatenate((stim_sides_arr_s[k],
                                            np.zeros(max_len_s-len(stim_sides_arr_s[k]))))
                            for k in range(len(stim_sides_arr_s))])
    session_uuids = np.array(session_uuids)

    # Fit models
    model_ns = exp_stimside('./results/', session_uuids, '%s_no_stim' % nickname,
                         actions_ns, stimuli_ns, stim_side_ns)
    model_ns.load_or_train(nb_steps=2000, remove_old=False) # put 2000 steps for biasedBayesian and smooth_stimside and 1000 for all others
    param_ns = model_ns.get_parameters(parameter_type='all') # if you want the parameters
    # compute prior (actions,  stimuli and stim_side have been passed as arguments to allow pseudo blocks)
    priors_ns, llk_ns, accuracy_ns = model_ns.compute_prior(actions_ns, stimuli_ns, stim_side_ns)


    model_s = exp_stimside('./results/', session_uuids, '%s_stim' % nickname,
                         actions_s, stimuli_s, stim_side_s)
    model_s.load_or_train(nb_steps=2000, remove_old=False) # put 2000 steps for biasedBayesian and smooth_stimside and 1000 for all others
    param_s = model_s.get_parameters(parameter_type='all') # if you want the parameters
    # compute prior (actions,  stimuli and stim_side have been passed as arguments to allow pseudo blocks)
    priors_s, llk_s, accuracy_s = model_s.compute_prior(actions_s, stimuli_s, stim_side_s)



    # Plot
    sns.set(context='talk', style='ticks', font_scale=1.5)
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(30, 30))



    sns.despine(trim=True)
    plt.tight_layout()
    plt.savefig(join(fig_path, '%s_opto_behavior' % nickname))