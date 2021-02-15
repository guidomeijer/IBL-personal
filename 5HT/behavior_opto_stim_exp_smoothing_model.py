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
from behavior_models import utils
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from my_functions import load_opto_trials, paths
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD_TRIALS = False
REMOVE_OLD_FIT = False
PRE_TRIALS = 5
POST_TRIALS = 11
_, fig_path, save_path = paths()
fig_path = join(fig_path, '5HT', 'opto-behavior')

subjects = pd.read_csv('subjects.csv')

results_df = pd.DataFrame()
block_switches = pd.DataFrame()
for i, nickname in enumerate(subjects['subject']):

    # Query sessions
    eids = one.search(subject=nickname, task_protocol='_iblrig_tasks_opto_biasedChoiceWorld')

    # Get trial data
    stimuli_arr_ns, actions_arr_ns, stim_sides_arr_ns, prob_left_ns, session_uuids = [], [], [], [], []
    stimuli_arr_s, actions_arr_s, stim_sides_arr_s, prob_left_s = [], [], [], []
    ses_count = 0
    for j, eid in enumerate(eids):
        try:
            data = utils.load_session(eid)
            if data['choice'] is not None and data['probabilityLeft'][0] == 0.5:
                laser_stimulation = one.load(eid, dataset_types=['_ibl_trials.laser_stimulation'])[0]
                stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
                stimuli_arr_ns.append(stimuli[laser_stimulation == 0])
                actions_arr_ns.append(actions[laser_stimulation == 0])
                stim_sides_arr_ns.append(stim_side[laser_stimulation == 0])
                prob_left_ns.append(data['probabilityLeft'][laser_stimulation == 0])
                stimuli_arr_s.append(stimuli[laser_stimulation == 1])
                actions_arr_s.append(actions[laser_stimulation == 1])
                stim_sides_arr_s.append(stim_side[laser_stimulation == 1])
                prob_left_s.append(data['probabilityLeft'][laser_stimulation == 1])
                session_uuids.append(eid)
        except Exception:
            print('Could not load trials for %s' % eid)

    # get maximum number of trials across sessions
    max_len_ns = np.array([len(stimuli_arr_ns[k]) for k in range(len(stimuli_arr_ns))]).max()
    max_len_s = np.array([len(stimuli_arr_s[k]) for k in range(len(stimuli_arr_s))]).max()

    # pad with 0 such that we obtain nd arrays of size nb_sessions x nb_trials and convert to arrays
    stimuli_ns = np.array([np.concatenate((stimuli_arr_ns[k], np.zeros(max_len_ns-len(stimuli_arr_ns[k]))))
                           for k in range(len(stimuli_arr_ns))])
    actions_ns = np.array([np.concatenate((actions_arr_ns[k], np.zeros(max_len_ns-len(actions_arr_ns[k]))))
                           for k in range(len(actions_arr_ns))])
    stim_side_ns = np.array([np.concatenate((stim_sides_arr_ns[k], np.zeros(max_len_ns-len(stim_sides_arr_ns[k]))))
                             for k in range(len(stim_sides_arr_ns))])
    stimuli_s = np.array([np.concatenate((stimuli_arr_s[k], np.zeros(max_len_s-len(stimuli_arr_s[k]))))
                          for k in range(len(stimuli_arr_ns))])
    actions_s = np.array([np.concatenate((actions_arr_s[k], np.zeros(max_len_s-len(actions_arr_s[k]))))
                          for k in range(len(actions_arr_s))])
    stim_side_s = np.array([np.concatenate((stim_sides_arr_s[k], np.zeros(max_len_s-len(stim_sides_arr_s[k]))))
                            for k in range(len(stim_sides_arr_s))])
    session_uuids = np.array(session_uuids)

    # Fit models
    model = exp_stimside('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                         actions_ns, stimuli_ns, stim_side_ns)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_ss = model.get_parameters(parameter_type='posterior_mean')
    priors_stimside = model.compute_prior(actions_ns, stimuli_ns, stim_side_ns)[0]
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_no_stim' % nickname,
                            actions_ns, stimuli_ns, stim_side_ns)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_pa = model.get_parameters(parameter_type='posterior_mean')
    priors_prevaction = model.compute_prior(actions_ns, stimuli_ns, stim_side_ns)[0]

    # Add tau to results dataframe
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'tau_ss': 1/param_ss[0],
                                                      'tau_pa': 1/param_pa[0],
                                                      'opto_stim': 'no stim',
                                                      'sert-cre': subjects.loc[i, 'sert-cre']}))

    # Add prior around block switch
    for k in range(len(priors_stimside)):
        transitions = np.array(np.where(np.diff(prob_left_ns[k]) != 0)[0]) + 1
        for t, trans in enumerate(transitions):
            block_switches = block_switches.append(pd.DataFrame(data={
                        'prior': priors_stimside[k][trans-PRE_TRIALS:trans+POST_TRIALS],
                        'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                        'change_to': prob_left_ns[k][trans],
                        'opto': 'stim',
                        'sert_cre': subjects.loc[i, 'sert-cre']}))

    model = exp_stimside('./model_fit_results/', session_uuids, '%s_stim' % nickname,
                         actions_s, stimuli_s, stim_side_s)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_ss = model.get_parameters(parameter_type='posterior_mean')
    priors_stimside = model.compute_prior(actions_s, stimuli_s, stim_side_s)[0]
    model = exp_prev_action('./model_fit_results/', session_uuids, '%s_stim' % nickname,
                            actions_s, stimuli_s, stim_side_s)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    param_pa = model.get_parameters(parameter_type='posterior_mean')
    results_df = results_df.append(pd.DataFrame(index=[len(results_df)],
                                                data={'tau_ss': 1/param_ss[0],
                                                      'tau_pa': 1/param_pa[0],
                                                      'opto_stim': 'stim',
                                                      'sert-cre': subjects.loc[i, 'sert-cre']}))

    # Add prior around block switch
    for k in range(len(priors_stimside)):
        transitions = np.array(np.where(np.diff(prob_left_s[k]) != 0)[0]) + 1
        for t, trans in enumerate(transitions):
            block_switches = block_switches.append(pd.DataFrame(data={
                        'prior': priors_stimside[k][trans-PRE_TRIALS:trans+POST_TRIALS],
                        'trial': np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS)),
                        'change_to': prob_left_s[k][trans],
                        'opto': 'no stim',
                        'sert_cre': subjects.loc[i, 'sert-cre']}))

# %% Plot
sns.set(context='talk', style='ticks', font_scale=1.5)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))


#results_df.groupby('sert-cre').plot(x='opto_stim', y='tau_ss', marker='o', ax=ax1)

sns.lineplot(x='opto_stim', y='tau_ss', hue='sert-cre', style='sert-cre', estimator=None,
             data=results_df, dashes=False, markers=True, ax=ax1)
ax1.set(xlabel='', ylabel='Lenght of integration window (tau)', ylim=[4, 16])

sns.lineplot(x='trial', y='prior', data=block_switches[block_switches['sert_cre'] == 1],
             hue='change_to', style='opto', palette='colorblind', ax=ax2, ci=68)
#plt.plot([0, 0], [0, 1], color=[0.5, 0.5, 0.5], ls='--')
handles, labels = ax2.get_legend_handles_labels()
labels = ['', 'Change to R', 'Change to L', '', 'Stim', 'No stim']
ax2.legend(handles, labels, frameon=False, prop={'size': 20})
ax2.set(ylabel='Prior', xlabel='Trials relative to block switch',
        title='Exponential smoothed previous stimulus side model')


sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(fig_path, 'model_opto_behavior'))