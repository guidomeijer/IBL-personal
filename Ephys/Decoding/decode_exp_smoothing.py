#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode from all brain regions
@author: Guido Meijer
"""

from os.path import join, isfile
import numpy as np
from brainbox.task import generate_pseudo_session
from brainbox.population import get_spike_counts_in_bins, regress
import pandas as pd
from scipy.stats import pearsonr
from behavior_models import utils
from sklearn.model_selection import KFold
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex, load_trials
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
import brainbox.io.one as bbone
from oneibl.one import ONE
from ibllib.atlas import BrainRegions
from brainbox.numerical import ismember
one = ONE()
br = BrainRegions()

# Settings
REMOVE_OLD_FIT = False
OVERWRITE = False
TARGET = 'prior-prevaction'  # block, stim-side. reward or choice
MIN_NEURONS = 5  # min neurons per region
DECODER = 'linear-regression'
VALIDATION = 'kfold'
INCL_NEURONS = 'all'  # all or pass-QC
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
ATLAS = 'beryl-atlas'
NUM_SPLITS = 5
CHANCE_LEVEL = 'other-trials'
ITERATIONS = 100  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DATA_PATH = join(DATA_PATH, 'Ephys', 'Decoding', DECODER)


# %% Initialize

def remap(ids, source='Allen', dest='Beryl'):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    return br.id[br.mappings[dest][inds]]

# Time windows
PRE_TIME = 0.2
POST_TIME = 0

# Query session list
eids, probes, subjects = query_sessions(selection=INCL_SESSIONS, return_subjects=True)

# Load in all trials
if CHANCE_LEVEL == 'other-trials':
    all_trials = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'all_trials.p'))

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
        data = utils.load_session(eid)
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
    priors = model.compute_prior(actions, stimuli, stim_side, parameter_type='posterior_mean')[0]

    # Now that we have the priors from the model fit, loop over sessions and decode
    for j, eid in enumerate(session_uuids):
        print('\nProcessing session %d of %d' % (j+1, len(session_uuids)))

        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                                                                        eid, aligned=True, one=one)
            trials = load_trials(eid)
        except Exception as error_message:
            print(error_message)
            continue

        # Check data integrity
        if check_trials(trials) is False:
            continue
        ssdf
        # Exclude 50/50 block
        trials = trials[trials['probabilityLeft'] != 0.5]

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

                # Decode prior from model fit
                times = np.column_stack(((trials.goCue_times - PRE_TIME),
                                         (trials.goCue_times + POST_TIME)))
                population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                            clus_region, times)
                population_activity = population_activity.T
                if VALIDATION == 'kfold-interleaved':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
                elif VALIDATION == 'kfold':
                    cv = KFold(n_splits=NUM_SPLITS, shuffle=False)
                if isinstance(priors[0], float):
                    these_priors = priors[:population_activity.shape[0]]
                else:
                    these_priors = priors[j][:population_activity.shape[0]]
                pred_prior, pred_prior_train = regress(population_activity, these_priors,
                                                       cross_validation=cv, return_training=True)
                r_prior = pearsonr(these_priors, pred_prior)[0]
                r_prior_train = pearsonr(these_priors, pred_prior_train)[0]

                # Decode block identity
                pred_block = regress(population_activity, trials['probabilityLeft'].values,
                                     cross_validation=cv)
                r_block = pearsonr(trials['probabilityLeft'].values, pred_block)[0]

                # Estimate chance level
                r_prior_null = np.empty(ITERATIONS)
                r_prior_train_null = np.empty(ITERATIONS)
                r_block_null = np.empty(ITERATIONS)
                for k in range(ITERATIONS):

                    # Null is pseudo sessions
                    if CHANCE_LEVEL == 'pseudo':
                        if TARGET == 'prior-stimside':
                            pseudo_trials = generate_pseudo_session(trials, generate_choices=False)
                            pseudo_trials['choice'] = np.nan
                        elif TARGET == 'prior-prevaction':
                            pseudo_trials = generate_pseudo_session(trials, generate_choices=True)
                        stim_side, stimuli, actions, prob_left = utils.format_data(pseudo_trials)
                        p_priors = model.compute_prior(np.array(actions), np.array(stimuli),
                                                       np.array(stim_side),
                                                       parameter_type='posterior_mean')[0]

                    # Null is behavior of other mice
                    elif CHANCE_LEVEL == 'other-trials':
                        # Exclude the current mice from all trials
                        all_trials_excl = all_trials[all_trials['subject'] != subject]

                        # Get a random chunck of trials the same length as the current session
                        null_selection = np.random.randint(all_trials_excl.shape[0])
                        while null_selection + trials.shape[0] > all_trials_excl.shape[0]:
                            null_selection = np.random.randint(all_trials_excl.shape[0])
                        null_trials = all_trials_excl[null_selection : (null_selection
                                                                        + trials.shape[0])]
                        stim_side, stimuli, actions, prob_left = utils.format_data(null_trials)
                        p_priors = model.compute_prior(np.array(actions), np.array(stimuli),
                                                       np.array(stim_side),
                                                       parameter_type='posterior_mean')[0]

                    # Decode prior of null trials
                    p_pred_prior, p_pred_prior_train = regress(population_activity,
                                                               p_priors,
                                                               cross_validation=cv,
                                                               return_training=True)
                    r_prior_null[k] = pearsonr(p_priors, p_pred_prior)[0]
                    r_prior_train_null[k] = pearsonr(p_priors, p_pred_prior_train)[0]

                    # Decode null block identity
                    p_pred_block = regress(population_activity, prob_left.values,
                                           cross_validation=cv)
                    r_block_null[k] = pearsonr(prob_left.values, p_pred_block)[0]


                # Add to dataframe
                decoding_result = decoding_result.append(pd.DataFrame(
                    index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                     'date': date,
                                     'eid': eid,
                                     'probe': probe,
                                     'region': region,
                                     'r_prior': r_prior,
                                     'r_prior_train': r_prior_train,
                                     'r_block': r_block,
                                     'r_prior_null': r_prior_null.mean(),
                                     'r_prior_train_null': r_prior_train_null.mean(),
                                     'r_block_null': r_block_null.mean(),
                                     'tau': 1 / params[0],
                                     'n_trials': trials.probabilityLeft.shape[0],
                                     'n_neurons': np.unique(clus_region).shape[0]}), sort=False)

        decoding_result.to_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s_%s-%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS,
                                                        int(PRE_TIME*1000), int(POST_TIME*1000)))))
