#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode from all brain regions
@author: Guido Meijer
"""

from os.path import join
import numpy as np
from brainbox.population import decode
from brainbox.task import differentiate_units
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import alf
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
TARGET = 'reward'  # block, stim-side. reward or choice
MIN_NEURONS = 5  # min neurons per region
DECODER = 'bayes-multinomial'
VALIDATION = 'kfold-interleaved'
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
NUM_SPLITS = 5
CHANCE_LEVEL = 'shuffle'  # pseudo-blocks, phase-rand, shuffle or none
ITERATIONS = 1000  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DOWNLOAD_TRIALS = False

# %% Initialize

# Time windows
if (TARGET == 'block') | (TARGET == 'block-first') | (TARGET == 'block-last'):
    PRE_TIME = 0.6
    POST_TIME = 0.1
elif TARGET == 'block-stim':
    PRE_TIME = 0
    POST_TIME = 0.3
elif TARGET == 'stim-side':
    MIN_CONTRAST = 0.2
    PRE_TIME = 0
    POST_TIME = 0.3
elif TARGET == 'reward':
    PRE_TIME = 0
    POST_TIME = 0.5
elif TARGET == 'choice':
    PRE_TIME = 0.2
    POST_TIME = 0

# Query session list
eids, probes = query_sessions(selection=INCL_SESSIONS)

# Initialize classifier
if DECODER == 'bayes-multinomial-no-prior':
    clf = MultinomialNB(fit_prior=False)
else:
    clf = DECODER

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

    # Get trial vectors based on decoding target
    if TARGET == 'block':
        incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
        trial_times = trials.stimOn_times[incl_trials]
        probability_left = trials.probabilityLeft[incl_trials]
        trial_ids = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)
    elif TARGET == 'stim-side':
        incl_trials = (((trials.contrastLeft > MIN_CONTRAST)
                        | (trials.contrastRight > MIN_CONTRAST))
                       & (trials.probabilityLeft != 0.5))
        trial_times = trials.stimOn_times[incl_trials]
        trial_ids = np.isnan(trials.contrastLeft[incl_trials]).astype(int)
    elif TARGET == 'reward':
        incl_trials = (trials.choice != 0) & (trials.probabilityLeft != 0.5)
        trial_times = trials.feedback_times[incl_trials]
        trial_ids = (trials.feedbackType[incl_trials] == -1).astype(int)
    elif TARGET == 'choice':
        incl_trials = (((trials.choice != 0) & (~np.isnan(trials.feedback_times)))
                       & (trials.probabilityLeft != 0.5))
        trial_times = trials.feedback_times[incl_trials]
        trial_ids = (trials.choice[incl_trials] == 1).astype(int)
        choice_ratio = ((np.sum(trial_ids == 0) - np.sum(trial_ids == 1))
                        / (np.sum(trial_ids == 0) + np.sum(trial_ids == 1)))
        if (choice_ratio > 0.95) or (choice_ratio < -0.95):
            print('Choices too biased, skipping session')
            continue

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
            print('Decoding region %s (%d of %d)' % (region, r + 1, len(np.unique(regions))))

            # Get clusters in this brain region
            region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
            clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]

            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                         clusters_in_region)]

            # Exclude neurons that drift over the session
            if INCL_NEURONS == 'no_drift':
                trial_half = np.zeros(trial_times.shape[0])
                trial_half[int(trial_half.shape[0]/2):] = 1
                drift_neurons = differentiate_units(spks_region, clus_region, trial_times,
                                                    trial_half, pre_time=PRE_TIME,
                                                    post_time=POST_TIME)[0]
                print('%d out of %d drift neurons detected' % (drift_neurons.shape[0],
                                                               np.unique(clus_region).shape[0]))
                spks_region = spks_region[~np.isin(clus_region, drift_neurons)]
                clus_region = clus_region[~np.isin(clus_region, drift_neurons)]

            # Check if there are enough neurons in this brain region
            if np.unique(clus_region).shape[0] < MIN_NEURONS:
                continue

            # Decode
            decode_result = decode(spks_region, clus_region, trial_times, trial_ids,
                                   pre_time=PRE_TIME, post_time=POST_TIME, classifier=clf,
                                   cross_validation=VALIDATION, num_splits=NUM_SPLITS)

            # Estimate chance level
            if CHANCE_LEVEL == 'phase-rand':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_ids,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=clf,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       phase_rand=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'shuffle':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_ids,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=clf,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       shuffle=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'pseudo-blocks':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_ids,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=clf,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       pseudo_blocks=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'none':
                decode_chance = []
            else:
                raise Exception('CHANCE_LEVEL must be phase_rand, shuffle or none')

            # Calculate p-values
            p_accuracy = (np.sum(decode_chance['accuracy'] > decode_result['accuracy'])
                          / decode_chance['accuracy'].shape[0])
            p_f1 = (np.sum(decode_chance['f1'] > decode_result['f1'])
                          / decode_chance['f1'].shape[0])
            p_auroc = (np.sum(decode_chance['auroc'] > decode_result['auroc'])
                          / decode_chance['auroc'].shape[0])

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                 'date': date,
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'f1': decode_result['f1'].mean(),
                                 'accuracy': decode_result['accuracy'].mean(),
                                 'auroc': decode_result['auroc'].mean(),
                                 'chance_accuracy': decode_chance['accuracy'].mean(),
                                 'chance_f1': decode_chance['f1'].mean(),
                                 'chance_auroc': decode_chance['auroc'].mean(),
                                 'p_accuracy': p_accuracy,
                                 'p_f1': p_f1,
                                 'p_auroc': p_auroc,
                                 'n_trials': trial_ids.shape[0],
                                 'n_neurons': np.unique(clus_region).shape[0]}))

    decoding_result.to_pickle(join(SAVE_PATH, DECODER,
                    ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                 INCL_SESSIONS, INCL_NEURONS))))

# Exclude root
decoding_result = decoding_result.reset_index()
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Drop duplicates
decoding_result = decoding_result[~decoding_result.duplicated(subset=['region', 'eid', 'probe'])]
