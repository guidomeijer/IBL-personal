#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode from all brain regions
@author: Guido Meijer
"""

from os.path import join
import numpy as np
from brainbox.population import _get_spike_counts_in_bins
from brainbox.task import generate_pseudo_session
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as sklearn_shuffle
import alf
from my_functions import paths, query_sessions, check_trials, combine_layers_cortex, classify
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
DECODER = 'bayes-multinomial'
VALIDATION = 'kfold-interleaved'
INCL_SESSIONS = 'aligned-behavior'  # all, aligned, resolved, aligned-behavior or resolved-behavior
NUM_SPLITS = 5
CHANCE_LEVEL = 'pseudo-session'  # pseudo-session, phase-rand, shuffle or none
ITERATIONS = 1000  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DOWNLOAD_TRIALS = True

# %% Initialize

# Query session list
eids, probes = query_sessions(selection=INCL_SESSIONS)

# Initialize classifier and cross-validation
if DECODER == 'bayes-multinomial':
    clf = MultinomialNB()
if VALIDATION == 'kfold-interleaved':
    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)


# %% MAIN
decoding_result = pd.DataFrame()
for i in range(len(eids)):
# for i in [1]:
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

    # Get trial times and ids
    incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_ids = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

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

            # Drop fiber tracts etc
            if region.islower():
                continue
            print('Decoding region %s (%d of %d)' % (region, r + 1, len(np.unique(regions))))

            if 'metrics' not in clusters[probe]:
                continue
            if clusters[probe]['acronym'].shape[0] != clusters[probe].metrics.cluster_id.shape[0]:
                continue

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

            # Get population response matrix of all trials
            dlc_matrix = []  ## INPUT MATRIX HERE

            # Decode
            accuracy = classify(dlc_matric, trial_ids, clf, cv)[0]

            null_iterations = np.empty(ITERATIONS)
            for k in range(ITERATIONS):
                # Estimate chance level
                if CHANCE_LEVEL == 'shuffle':
                    null_iterations[k] = classify(dlc_matrix, sklearn_shuffle(trial_ids),
                                                  clf, cv)[0]
                elif CHANCE_LEVEL == 'pseudo-session':
                    pseudo_trials = generate_pseudo_session(trials)
                    pseudo_incl = (pseudo_trials.probabilityLeft == 0.8) | (pseudo_trials.probabilityLeft == 0.2)
                    trial_times = pseudo_trials.stimOn_times[pseudo_incl]
                    probability_left = pseudo_trials.probabilityLeft[pseudo_incl]
                    pseudo_trial_ids = (pseudo_trials.probabilityLeft[pseudo_incl] == 0.2).astype(int)
                    null_iterations[k] = classify(dlc_matrix, pseudo_trial_ids, clf, cv)[0]
                elif CHANCE_LEVEL == 'none':
                    null_iterations = []
                else:
                    raise Exception('CHANCE_LEVEL must be phase_rand, shuffle or none')

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[decoding_result.shape[0] + 1], data={'subject': subject,
                                 'date': date, 'eid': eid, 'probe': probe, 'region': region,
                                 'accuracy': accuracy,
                                 'chance_accuracy': null_iterations.mean(),
                                 'n_trials': np.sum(incl_trials),
                                 'iterations': ITERATIONS,
                                 'n_neuron_pick': N_NEURON_PICK}))

    decoding_result.to_pickle(join(SAVE_PATH, DECODER,
                    ('%s_%s_%s_%s_%s_cells.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                                 INCL_SESSIONS, N_NEURONS))))

# Drop duplicates
decoding_result = decoding_result.reset_index()
decoding_result = decoding_result[~decoding_result.duplicated(subset=['region', 'eid', 'probe'])]
