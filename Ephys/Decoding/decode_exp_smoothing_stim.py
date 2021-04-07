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
OVERWRITE = True
TARGET = 'prior-prevaction-stim'
MIN_NEURONS = 5  # min neurons per region
DECODER = 'linear-regression'
VALIDATION = 'kfold'
INCL_NEURONS = 'all'  # all or pass-QC
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
ATLAS = 'beryl-atlas'
NUM_SPLITS = 5
CHANCE_LEVEL = 'pseudo'  # pseudo, phase-rand, shuffle or none
ITERATIONS = 0  # for null-distribution
N_TRIAL_PICKS = 50  # for trial selection
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DATA_PATH = join(DATA_PATH, 'Ephys', 'Decoding', DECODER)


# %% Initialize

def remap(ids, source='Allen', dest='Beryl'):
    _, inds = ismember(ids, br.id[br.mappings[source]])
    return br.id[br.mappings[dest][inds]]

# Time windows
PRE_TIME = 0
POST_TIME = 0.3

# Query session list
eids, probes, subjects = query_sessions(selection=INCL_SESSIONS, return_subjects=True)


def balanced_trial_set(trials):
    # A balanced random subset of left and right trials from the two blocks
    left_stim = (trials.contrastLeft > 0) & (trials.probabilityLeft == 0.2)
    left_stim[np.random.choice(np.where(((trials.contrastLeft > 0)
                                         & (trials.probabilityLeft == 0.8)))[0],
                               size=np.sum(left_stim), replace=False)] = True
    right_stim = (trials.contrastRight > 0) & (trials.probabilityLeft == 0.8)
    right_stim[np.random.choice(np.where(((trials.contrastRight > 0)
                                         & (trials.probabilityLeft == 0.2)))[0],
                               size=np.sum(right_stim), replace=False)] = True
    incl_trials = (left_stim | right_stim
                   | (((trials.contrastLeft == 0) | (trials.contrastRight == 0))
                      & (trials.probabilityLeft != 0.5)))
    return incl_trials


# %% MAIN

# Load in decoding done so far if required
if OVERWRITE:
    decoding_result = pd.DataFrame(columns=['subject', 'date', 'eid', 'probe', 'region'])
elif isfile(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS)))):
    decoding_result = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS))))
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
    if 'prior-stimside' in TARGET:
        model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                             session_uuids, subject, actions, stimuli, stim_side)
    elif 'prior-prevaction' in TARGET:
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
                r_prior, r_prior_train = np.empty(N_TRIAL_PICKS), np.empty(N_TRIAL_PICKS)
                r_prior_pseudo, r_prior_train_pseudo = np.empty(N_TRIAL_PICKS), np.empty(N_TRIAL_PICKS)
                for n in range(N_TRIAL_PICKS):

                    # Select trials which are balanced on left and right
                    incl_trials = balanced_trial_set(trials)

                    times = np.column_stack(((trials.goCue_times[incl_trials] - PRE_TIME),
                                             (trials.goCue_times[incl_trials] + POST_TIME)))
                    population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                                clus_region, times)
                    population_activity = population_activity.T
                    if VALIDATION == 'kfold-interleaved':
                        cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
                    elif VALIDATION == 'kfold':
                        cv = KFold(n_splits=NUM_SPLITS, shuffle=False)
                    if isinstance(priors[0], float):
                        these_priors = priors[incl_trials]
                    else:
                        these_priors = priors[j][:trials.shape[0]][incl_trials]
                    pred_prior, pred_prior_train = regress(population_activity, these_priors,
                                                           cross_validation=cv, return_training=True)
                    r_prior[n] = pearsonr(these_priors, pred_prior)[0]
                    r_prior_train[n] = pearsonr(these_priors, pred_prior_train)[0]

                    # Estimate chance level
                    r_pseudo, r_train_pseudo = np.empty(ITERATIONS), np.empty(ITERATIONS)
                    for k in range(ITERATIONS):
                        if 'prior-stimside' in TARGET:
                            pseudo_trials = generate_pseudo_session(trials, generate_choices=False)
                            pseudo_trials['choice'] = np.nan
                        elif 'prior-prevaction' in TARGET:
                            pseudo_trials = generate_pseudo_session(trials, generate_choices=True)
                        stim_side, stimuli, actions, prob_left = utils.format_data(pseudo_trials)
                        p_priors = model.compute_prior(np.array(actions), np.array(stimuli),
                                                       np.array(stim_side),
                                                       parameter_type='posterior_mean')[0]

                        # Decode prior
                        p_pred_prior, p_pred_prior_train = regress(population_activity, p_priors,
                                                                   cross_validation=cv, return_training=True)
                        r_pseudo[k] = pearsonr(p_priors, p_pred_prior)[0]
                        r_train_pseudo[k] = pearsonr(p_priors, p_pred_prior_train)[0]

                    # Get mean over iterations
                    if ITERATIONS > 0:
                        r_prior_pseudo[n] = r_pseudo.mean()
                        r_prior_train_pseudo[n] = r_train_pseudo.mean()
                    else:
                        r_prior_pseudo, r_prior_train_pseudo = np.nan, np.nan

                # Add to dataframe
                decoding_result = decoding_result.append(pd.DataFrame(
                    index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                     'date': date,
                                     'eid': eid,
                                     'probe': probe,
                                     'region': region,
                                     'r_prior': np.mean(r_prior),
                                     'r_prior_train': np.mean(r_prior_train),
                                     'r_prior_null': np.mean(r_prior_pseudo),
                                     'r_prior_train_null': np.mean(r_prior_train_pseudo),
                                     'tau': 1 / params[0],
                                     'n_trials': trials.probabilityLeft.shape[0],
                                     'n_trials_balanced': incl_trials.sum(),
                                     'n_neurons': np.unique(clus_region).shape[0]}))

        decoding_result.to_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                        ('%s_%s_%s_%s_%s_cells_%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                        INCL_SESSIONS, INCL_NEURONS, ATLAS))))
