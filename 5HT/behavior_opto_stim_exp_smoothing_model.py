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
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from my_functions import load_opto_trials, paths
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD_TRIALS = False
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

results_df = pd.DataFrame()
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
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                         actions_ns, stimuli_ns, stim_side_ns)
    model.load_or_train(nb_steps=2000, remove_old=False)
    param_ss = model.get_parameters(parameter_type='posterior_mean')
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                            actions_ns, stimuli_ns, stim_side_ns)
    model.load_or_train(nb_steps=2000, remove_old=False)
    param_pa = model.get_parameters(parameter_type='posterior_mean')
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'tau_ss': 1/param_ss[0],
                                                      'tau_pa': 1/param_pa[0],
                                                      'opto_stim': 'no stim',
                                                      'sert-cre': subjects.loc[i, 'sert-cre']}))


    model = exp_stimside('./model_fit_results/', session_uuids, '%s_stim' % nickname,
                         actions_s, stimuli_s, stim_side_s)
    model.load_or_train(nb_steps=2000, remove_old=False)
    param_ss = model.get_parameters(parameter_type='posterior_mean')
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_stim' % nickname,
                            actions_s, stimuli_s, stim_side_s)
    model.load_or_train(nb_steps=2000, remove_old=False)
    param_pa = model.get_parameters(parameter_type='posterior_mean')
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'tau_ss': 1/param_ss[0],
                                                      'tau_pa': 1/param_pa[0],
                                                      'opto_stim': 'stim',
                                                      'sert-cre': subjects.loc[i, 'sert-cre']}))

# Plot
sns.set(context='talk', style='ticks', font_scale=1.5)
f, ax1 = plt.subplots(1, 1, figsize=(10, 10))

sns.lineplot(x='opto_stim', y='tau_pa', hue='sert-cre', estimator=None, data=results_df)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_opto_behavior'))