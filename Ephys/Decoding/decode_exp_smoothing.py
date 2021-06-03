# %%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode from all brain regions
@author: Guido Meijer
"""

from os.path import join, isfile
import numpy as np
from brainbox.task.closed_loop import generate_pseudo_session
from brainbox.population.decode import get_spike_counts_in_bins, regress
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex, load_trials
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
import brainbox.io.one as bbone
from oneibl.one import ONE
from ibllib.atlas import BrainRegions
from sklearn.metrics import mean_squared_error
from brainbox.numerical import ismember
one = ONE()
br = BrainRegions()

# Settings
REMOVE_OLD_FIT = False
OVERWRITE = False
TARGET = 'prior-prevaction'
MIN_NEURONS = 5  # min neurons per region
REGULARIZATION = 'L2'
DECODER = 'linear-regression-%s' % REGULARIZATION
VALIDATION = 'kfold'
INCL_NEURONS = 'all'  # all or pass-QC
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
ATLAS = 'beryl-atlas'
NUM_SPLITS = 5
CHANCE_LEVEL = 'other-trials'
ITERATIONS = 20  # for null distribution estimation
PRE_TIME = .6
POST_TIME = -.1
MIN_RT = 0.08  # in seconds
EXCL_5050 = True
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DATA_PATH = join(DATA_PATH, 'Ephys', 'Decoding', DECODER)

# %% Initialize


def remap(ids, source='Allen', dest='Beryl'):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    return br.id[br.mappings[dest][inds]]


def get_incl_trials(trials, target, excl_5050, min_rt):
    incl_trials = np.ones(trials.shape[0]).astype(bool)
    if excl_5050:
        incl_trials[trials['probabilityLeft'] == 0.5] = False
    if 'pos' in target:
        incl_trials[trials['feedbackType'] == -1] = False  # Exclude all rew. ommissions
    if 'neg' in target:
        incl_trials[trials['feedbackType'] == 1] = False  # Exclude all rewards
    if '0' in target:
        incl_trials[trials['signed_contrast'] != 0] = False  # Only include 0% contrast
    if ('prior' in target) and ('stim' in target):
        incl_trials[trials['signed_contrast'] != 0] = False  # Only include 0% contrast
    incl_trials[trials['reaction_times'] < min_rt] = False  # Exclude trials with fast rt
    return incl_trials


# Query session list
eids, probes, subjects = query_sessions(selection=INCL_SESSIONS, return_subjects=True)

# Load in all trials
if CHANCE_LEVEL == 'other-trials':
    all_trials = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'all_trials.p'))
all_trials = all_trials[get_incl_trials(all_trials, TARGET, EXCL_5050, MIN_RT)]  # trial selection

# %% MAIN

# Load in decoding done so far if required
if OVERWRITE:
    decoding_result = pd.DataFrame(columns=['subject', 'date', 'eid', 'probe', 'region'])
elif isfile(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s_%s-%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS,
                                                        int(PRE_TIME*1000), int(POST_TIME*1000))))):
    decoding_result = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s_%s-%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS,
                                                        int(PRE_TIME*1000), int(POST_TIME*1000)))))
else:
    decoding_result = pd.DataFrame(columns=['subject', 'date', 'eid', 'probe', 'region'])

# Loop over subjects
for i, subject in enumerate(np.unique(subjects)):
    print('\nStarting subject %s [%d of %d]\n' % (subject, i + 1, len(np.unique(subjects))))

    # Generate stimulus vectors for all sessions of this subject
    stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
    for j, eid in enumerate(eids[subjects == subject]):
        try:
            # Load in trials vectors
            trials = load_trials(eid, invert_stimside=True, one=one)
            incl_trials = get_incl_trials(trials, TARGET, EXCL_5050, MIN_RT)
            stimuli_arr.append(trials['signed_contrast'][incl_trials].values)
            actions_arr.append(trials['choice'][incl_trials].values)
            stim_sides_arr.append(trials['stim_side'][incl_trials].values)
            session_uuids.append(eid)
        except:
            print(f'Could not load trials for {eid}')
    print(f'\nLoaded data from {len(session_uuids)} sessions')
    if len(session_uuids) == 0:
        continue

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
    if 'stimside' in TARGET:
        model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                             session_uuids, subject, actions, stimuli, stim_side)
    elif 'prevaction' in TARGET:
        model = exp_prev_action(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                                session_uuids, subject, actions, stimuli, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    params = model.get_parameters(parameter_type='posterior_mean')

    if 'prior' in TARGET:
        target = model.compute_signal(signal='prior', act=actions, stim=stimuli, side=stim_side,
                                      parameter_type='posterior_mean', verbose=False)['prior']
    elif 'prederr' in TARGET:
        target = model.compute_signal(signal='prediction_error', act=actions, stim=stimuli,
                                      side=stim_side, verbose=False,
                                      parameter_type='posterior_mean')['prediction_error']
    target = np.squeeze(np.array(target))

    # Make target absolute
    if 'abs' in TARGET:
        target = np.abs(target)

    # Now that we have the priors from the model fit, loop over sessions and decode
    for j, eid in enumerate(session_uuids):
        print('\nProcessing session %d of %d' % (j+1, len(session_uuids)))

        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                                                                        eid, aligned=True, one=one)
            trials = load_trials(eid)
            trials = trials[get_incl_trials(trials, TARGET, EXCL_5050, MIN_RT)]
        except Exception as error_message:
            print(error_message)
            continue

        # Check data integrity
        if check_trials(trials) is False:
            continue

        # Get trial triggers
        if 'prior' in TARGET:
            trial_times = trials.goCue_times
        elif 'prederr' in TARGET:
            trial_times = trials.feedback_times

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

            # Check if cluster metrics are available
            if 'metrics' not in clusters[probe]:
                continue

            # Get list of brain regions
            if ATLAS == 'beryl-atlas':
                mapped_br = br.get(ids=remap(clusters[probe]['atlas_id']))
                clusters_regions = mapped_br['acronym']

            elif ATLAS == 'allen-atlas':
                clusters_regions = combine_layers_cortex(clusters[probe]['acronym'])

            # Get list of neurons that pass QC
            if INCL_NEURONS == 'pass-QC':
                clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
            elif INCL_NEURONS == 'all':
                clusters_pass = np.arange(clusters[probe]['metrics'].shape[0])

            # Decode per brain region
            for r, region in enumerate(np.unique(clusters_regions)):

                # Skip region if any of these conditions apply
                if region.islower():
                    continue

                if (OVERWRITE == False) and (np.sum((decoding_result['eid'] == eid)
                                                    & (decoding_result['probe'] == probe)
                                                    & (decoding_result['region'] == region)) != 0):
                    print('Region %s already decoded, skipping..' % region)
                    continue
                print('Decoding region %s (%d of %d)' % (region, r + 1, len(np.unique(clusters_regions))))

                # Get clusters in this brain region
                clusters_in_region = [x for x, y in enumerate(clusters_regions)
                                      if (region == y) and (x in clusters_pass)]

                # Select spikes and clusters
                spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
                clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                             clusters_in_region)]

                # Check if there are enough neurons in this brain region
                if np.unique(clus_region).shape[0] < MIN_NEURONS:
                    continue

                # Get population activity for all trials
                times = np.column_stack(((trial_times - PRE_TIME),
                                         (trial_times + POST_TIME)))
                population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                            clus_region, times)
                population_activity = population_activity.T

                # Subtract mean firing rates for all stim types
                if 'norm' in TARGET:
                    norm_pop = np.empty(population_activity.shape)
                    for s, contrast in enumerate(trials['signed_contrast']):
                        norm_pop[s, :] = (population_activity[s, :]
                                          - np.mean(population_activity[trials['signed_contrast'] == contrast, :], axis=0))
                    population_activity = norm_pop

                # Initialize cross-validation
                if VALIDATION == 'kfold-interleaved':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
                elif VALIDATION == 'kfold':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=False)

                # Get target to use for this session
                if len(target.shape) == 1:
                    this_target = target
                elif len(target.shape) == 2:
                    this_target = target[j, :trials.shape[0]]

                # Decode selected trials
                pred_target, pred_target_train = regress(population_activity,
                                                         this_target, cross_validation=cv,
                                                         return_training=True,
                                                         regularization=REGULARIZATION)
                r_target = pearsonr(this_target, pred_target)[0]
                mse_target = mean_squared_error(this_target, pred_target)
                r_target_train = pearsonr(this_target, pred_target_train)[0]
                mse_target_train = mean_squared_error(this_target, pred_target_train)

                # Estimate chance level
                r_null = np.empty(ITERATIONS)
                r_train_null = np.empty(ITERATIONS)
                mse_null = np.empty(ITERATIONS)
                mse_train_null = np.empty(ITERATIONS)
                for k in range(ITERATIONS):

                    # Null is pseudo sessions
                    if CHANCE_LEVEL == 'pseudo':
                        if 'stimside' in TARGET:
                            null_trials = generate_pseudo_session(trials, generate_choices=False)
                            null_trials['choice'] = np.nan
                        elif 'prevaction' in TARGET:
                            null_trials = generate_pseudo_session(trials, generate_choices=True)
                        null_trials['signed_contrast'] = -null_trials['signed_contrast']

                    # Null is behavior of other mice
                    elif CHANCE_LEVEL == 'other-trials':
                        # Exclude the current mice from all trials
                        all_trials_excl = all_trials[all_trials['subject'] != subject]

                        # Get a random chunck of trials the same length as the current session
                        null_selection = np.random.randint(all_trials_excl.shape[0])
                        while null_selection + trials.shape[0] >= all_trials_excl.shape[0]:
                            null_selection = np.random.randint(all_trials_excl.shape[0])
                        null_trials = all_trials_excl[null_selection : (null_selection
                                                                        + trials.shape[0])]

                    # Get null target
                    if 'prior' in TARGET:
                        signal = 'prior'
                    elif 'prederr' in TARGET:
                        signal = 'prediction_error'
                    null_target = model.compute_signal(signal=signal,
                                                       act=null_trials['choice'].values,
                                                       stim=null_trials['signed_contrast'].values,
                                                       side=null_trials['stim_side'].values,
                                                       parameter_type='posterior_mean',
                                                       verbose=False)[signal]
                    null_target = np.squeeze(np.array(null_target))

                    if 'abs' in TARGET:
                        null_target = np.abs(null_target)

                    # Decode prior of null trials
                    null_pred, null_pred_train = regress(population_activity, null_target,
                                                         cross_validation=cv, return_training=True,
                                                         regularization=REGULARIZATION)
                    r_null[k] = pearsonr(null_target, null_pred)[0]
                    mse_null[k] = mean_squared_error(null_target, null_pred)
                    r_train_null[k] = pearsonr(null_target, null_pred_train)[0]
                    mse_train_null[k] = mean_squared_error(null_target, null_pred_train)

                # Add to dataframe
                decoding_result = decoding_result.append(pd.DataFrame(
                    index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                     'date': date,
                                     'eid': eid,
                                     'probe': probe,
                                     'region': region,
                                     'r': r_target,
                                     'mse': mse_target,
                                     'r_train': r_target_train,
                                     'mse_train': mse_target_train,
                                     'r_null': r_null.mean(),
                                     'mse_null': mse_null.mean(),
                                     'r_train_null': r_train_null.mean(),
                                     'mse_train_null': mse_train_null.mean(),
                                     'p_value_r': np.sum(r_target > r_null) / len(r_null),
                                     'p_value_mse': np.sum(mse_target > mse_null) / len(mse_null),
                                     'tau': 1 / params[0],
                                     'n_trials': trials.probabilityLeft.shape[0],
                                     'n_neurons': np.unique(clus_region).shape[0]}), sort=False)

        decoding_result.to_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s_%s-%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS,
                                                        int(PRE_TIME*1000), int(POST_TIME*1000)))))
