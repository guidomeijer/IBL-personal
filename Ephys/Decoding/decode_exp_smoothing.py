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
from brainbox.population import regress
import pandas as pd
import alf
from scipy.stats import pearsonr
import warnings
from prior_funcs import perform_inference
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex, figure_style
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Settings
REMOVE_OLD_FIT = True
TARGET = 'exp-smoothing'  # block, stim-side. reward or choice
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
eids, probes = query_sessions(selection=INCL_SESSIONS)

# %% MAIN
decoding_result = pd.DataFrame()
for i in range(len(eids)):
    print('\nProcessing session %d of %d' % (i+1, len(eids)))

    # Load in data
    eid = eids[i]
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
    probes_to_use = probes[i]

    # Fit exponential smoothing model
    actions = trials['choice'] + 2
    actions[actions == 3] = -1
    signed_contrast = trials['contrastRight'].copy()
    signed_contrast[np.isnan(signed_contrast)] = -trials['contrastLeft'][
                                                            ~np.isnan(trials['contrastLeft'])]
    stim_side = (signed_contrast > 0).astype(int)
    stim_side[stim_side == 0]= -1
    stim_side[(signed_contrast == 0) & (np.isnan(trials['contrastLeft']))] = 1
    stim_side[(signed_contrast == 0) & (np.isnan(trials['contrastRight']))] = 1
    model = exp_prev_action(join(SAVE_PATH, 'Ephys', 'behavior_model_results/'), [eid],
                         eid, actions, signed_contrast, stim_side)
    model.load_or_train(nb_steps=2000, remove_old=REMOVE_OLD_FIT)
    params = model.get_parameters(parameter_type='posterior_mean')
    priors, llk, accuracy = model.compute_prior(actions, signed_contrast, stim_side)

    sdf

    stim_side = (np.array(np.isnan(trials.contrastLeft)==False) * -1
                 + np.array(np.isnan(trials.contrastRight)==False)) * 1
    infer_p_left = perform_inference(stim_side)[0]
    infer_p_left = infer_p_left[:, 0]

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

            # Decode inferred pLeft
            times = np.column_stack(((trials.goCue_times - PRE_TIME),
                                     (trials.goCue_times + POST_TIME)))
            population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                        clus_region, times)
            population_activity = population_activity.T
            regress()

            pred_pleft = linear_regression(spks_region, clus_region, trials['stimOn_times'],
                                           infer_p_left, pre_time=PRE_TIME, post_time=POST_TIME,
                                           cross_validation='kfold-interleaved')
            pred_pleft_r = pearsonr(pred_pleft, infer_p_left)[0]

            # Decode block
            pred_block = linear_regression(spks_region, clus_region, trials['stimOn_times'],
                                           trials.probabilityLeft, pre_time=PRE_TIME,
                                           post_time=POST_TIME,
                                           cross_validation='kfold-interleaved')
            pred_block_r = pearsonr(pred_block, trials.probabilityLeft)[0]

            # Estimate chance level
            pred_pseudo_pleft_r = np.empty(ITERATIONS)
            pred_pseudo_block_r = np.empty(ITERATIONS)
            for j in range(ITERATIONS):

                # Generate pseudo session
                pseudo_trials = generate_pseudo_session(trials)
                pseudo_stim_side = (np.array(np.isnan(pseudo_trials.contrastLeft)==False) * -1
                                    + np.array(np.isnan(pseudo_trials.contrastRight)==False)) * 1
                pseudo_infer_p_left = perform_inference(pseudo_stim_side)[0]
                pseudo_infer_p_left = pseudo_infer_p_left[:, 0]

                # Decode pseudo inferred pleft
                pred_pseudo_pleft = linear_regression(spks_region, clus_region,
                                                      trials['stimOn_times'],
                                                      pseudo_infer_p_left,
                                                      pre_time=PRE_TIME, post_time=POST_TIME,
                                                      cross_validation='kfold-interleaved')
                pred_pseudo_pleft_r[j] = pearsonr(pred_pseudo_pleft, pseudo_infer_p_left)[0]

                # Decode pseudo block
                pred_pseudo_block = linear_regression(spks_region, clus_region,
                                                      trials['stimOn_times'],
                                                      pseudo_trials.probabilityLeft,
                                                      pre_time=PRE_TIME, post_time=POST_TIME,
                                                      cross_validation='kfold-interleaved')
                pred_pseudo_block_r[j] = pearsonr(pred_pseudo_pleft,
                                                  pseudo_trials.probabilityLeft)[0]

            # Calculate p values
            p_pleft = np.sum(pred_pseudo_pleft_r > pred_pleft_r) / pred_pseudo_pleft_r.shape[0]
            p_block = np.sum(pred_pseudo_block_r > pred_pleft_r) / pred_pseudo_block_r.shape[0]

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                 'date': date,
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'r_infer': pred_pleft_r,
                                 'r_block': pred_block_r,
                                 'r_infer_null': pred_pseudo_pleft_r.mean(),
                                 'r_block_null': pred_pseudo_block_r.mean(),
                                 'p_infer': p_pleft,
                                 'p_block': p_block,
                                 'n_trials': trials.probabilityLeft.shape[0],
                                 'n_neurons': np.unique(clus_region).shape[0]}))

    decoding_result.to_pickle(join(SAVE_PATH, DECODER,
                    ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                 INCL_SESSIONS, INCL_NEURONS))))

# Drop duplicates
decoding_result = decoding_result.reset_index()
decoding_result = decoding_result[~decoding_result.duplicated(subset=['region', 'eid', 'probe'])]
