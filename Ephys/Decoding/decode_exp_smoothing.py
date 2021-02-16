#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode from all brain regions
@author: Guido Meijer
"""

from os.path import join
import numpy as np
from brainbox.task import generate_pseudo_session
from brainbox.population import get_spike_counts_in_bins, regress
import pandas as pd
import alf
from scipy.stats import pearsonr
import warnings
from behavior_models import utils
from sklearn.model_selection import KFold
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Settings
REMOVE_OLD_FIT = False
TARGET = 'prior-stimside'  # block, stim-side. reward or choice
MIN_NEURONS = 5  # min neurons per region
DECODER = 'linear-regression'
VALIDATION = 'kfold-interleaved'
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
NUM_SPLITS = 5
CHANCE_LEVEL = 'pseudo'  # pseudo, phase-rand, shuffle or none
ITERATIONS = 100  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DOWNLOAD_TRIALS = False

# %% Initialize

# Time windows
PRE_TIME = 0.6
POST_TIME = -0.1

# Query session list
eids, probes, subjects = query_sessions(selection=INCL_SESSIONS, return_subjects=True)

# %% MAIN

decoding_result = pd.DataFrame()
for i, subject in enumerate(np.unique(subjects)):
    print('\nStarting subject %s [%d of %d]\n' % (subject, i + 1, len(np.unique(subjects))))

    # Generate stimulus vectors for all sessions of this subject
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
    if TARGET == 'prior-stimside':
        model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                             session_uuids, subject, actions, stimuli, stim_side)
    elif TARGET == 'prior-prevaction':
        model = exp_prev_action(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                                session_uuids, subject, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    params = model.get_parameters(parameter_type='posterior_mean')
    priors = model.compute_prior(actions, stimuli, stim_side)[0]

    # Now that we have the priors from the model fit, loop over sessions and decode
    for j, eid in enumerate(session_uuids):
        print('\nProcessing session %d of %d' % (j+1, len(session_uuids)))

        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                                                                        eid, aligned=True, one=one)
            ses_path = one.path_from_eid(eid)
            if DOWNLOAD_TRIALS:
                _ = one.load(eid, dataset_types=['trials.stimOn_times', 'trials.probabilityLeft',
                                                 'trials.contrastLeft', 'trials.contrastRight',
                                                 'trials.feedbackType', 'trials.choice',
                                                 'trials.feedback_times'],
                             download_only=True, clobber=True)
            trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
        except Exception as error_message:
            print(error_message)
            continue

        # Check data integrity
        if check_trials(trials) is False:
            continue

        # Extract session data
        ses_info = one.get_details(eid)
        subject = ses_info['subject']
        date = ses_info['start_time'][:10]
        probes_to_use = probes[np.where(eids == eid)[0][0]]

         # Decode per brain region
        for p, probe in enumerate(probes_to_use):
            print('Processing %s (%d of %d)' % (probe, p + 1, len(probes_to_use)))

            # Check if data is available for this probe
            if probe not in clusters.keys():
                continue

            # Check if histology is available for this probe
            if not hasattr(clusters[probe], 'acronym'):
                continue

            # Get brain regions and combine cortical layers
            regions = combine_layers_cortex(np.unique(clusters[probe]['acronym']))

            # Decode per brain region
            for r, region in enumerate(np.unique(regions)):
                if region.islower():
                    continue
                if 'metrics' not in clusters[probe]:
                    continue
                print('Decoding region %s (%d of %d)' % (region, r + 1, len(np.unique(regions))))

                # Get clusters in this brain region
                region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
                clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]

                # Select spikes and clusters
                spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
                clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                             clusters_in_region)]

                # Check if there are enough neurons in this brain region
                if np.unique(clus_region).shape[0] < MIN_NEURONS:
                    continue

                # Decode prior from model fit
                times = np.column_stack(((trials.goCue_times - PRE_TIME),
                                         (trials.goCue_times + POST_TIME)))
                population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                            clus_region, times)
                population_activity = population_activity.T
                if VALIDATION == 'kfold-interleaved':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
                pred_prior = regress(population_activity, priors[j][:population_activity.shape[0]],
                                     cross_validation=cv)
                r_prior = pearsonr(priors[j][:population_activity.shape[0]],
                                   pred_prior)[0]

                # Decode block identity
                pred_block = regress(population_activity, trials['probabilityLeft'],
                                     cross_validation=cv)
                r_block = pearsonr(priors[j][:population_activity.shape[0]], pred_block)[0]

                # Estimate chance level
                r_prior_pseudo = np.empty(ITERATIONS)
                r_block_pseudo = np.empty(ITERATIONS)
                for k in range(ITERATIONS):
                    pseudo_trials = generate_pseudo_session(trials)
                    p_stim_side, p_stimuli, p_actions, _ = utils.format_data(pseudo_trials)
                    p_all_actions = actions.copy()
                    p_all_stimuli = stimuli.copy()
                    p_all_stim_side = stim_side.copy()
                    p_all_actions[j][:p_actions.shape[0]] = p_actions
                    p_all_stimuli[j][:p_stimuli.shape[0]] = p_stimuli
                    p_all_stim_side[j][:p_stim_side.shape[0]] = p_stim_side
                    if TARGET == 'prior-stimside':
                        p_model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                                               session_uuids, subject, p_all_actions,
                                               p_all_stimuli, p_all_stim_side)
                    elif TARGET == 'prior-prevaction':
                        p_model = exp_prev_action(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                                                  session_uuids, subject, p_all_actions,
                                                  p_all_stimuli, p_all_stim_side)
                    p_model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
                    p_priors = p_model.compute_prior(p_all_actions, p_all_stimuli,
                                                     p_all_stim_side)[0]

                    # Decode pseudo prior
                    p_pred_prior = regress(population_activity,
                                           p_priors[j][:population_activity.shape[0]],
                                           cross_validation=cv)
                    r_prior_pseudo[k] = pearsonr(p_priors[j][:population_activity.shape[0]],
                                                 p_pred_prior)[0]

                    # Decode pseudo block identity
                    p_pred_block = regress(population_activity, pseudo_trials['probabilityLeft'],
                                           cross_validation=cv)
                    r_block_pseudo[k] = pearsonr(pseudo_trials['probabilityLeft'], p_pred_block)[0]

                # Calculate p values
                p_prior = np.sum(r_prior_pseudo > r_prior) / r_prior_pseudo.shape[0]
                p_block = np.sum(r_block_pseudo > r_block) / r_block_pseudo.shape[0]

                # Add to dataframe
                decoding_result = decoding_result.append(pd.DataFrame(
                    index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                     'date': date,
                                     'eid': eid,
                                     'probe': probe,
                                     'region': region,
                                     'r_prior': r_prior,
                                     'r_block': r_block,
                                     'r_prior_null': r_prior_pseudo.mean(),
                                     'r_block_null': r_block_pseudo.mean(),
                                     'p_prior': p_prior,
                                     'p_block': p_block,
                                     'tau': 1 / params[0],
                                     'n_trials': trials.probabilityLeft.shape[0],
                                     'n_neurons': np.unique(clus_region).shape[0]}))

        decoding_result.to_pickle(join(SAVE_PATH, DECODER,
                        ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                     INCL_SESSIONS, INCL_NEURONS))))
