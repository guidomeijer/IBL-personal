#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020
Decode left/right block identity from all brain regions
@author: Guido Meijer
"""

from os.path import join
import numpy as np
from brainbox.population import decode
from brainbox.task import differentiate_units
import pandas as pd
import alf
from ephys_functions import paths, query_sessions, check_trials, combine_layers_cortex
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
PRE_TIME = 0.6
POST_TIME = -0.1
MIN_NEURONS = 5  # min neurons per region
MIN_TRIALS = 300
DECODER = 'bayes'
VALIDATION = 'kfold'
EXCL_DRIFT_NEURONS = True
NUM_SPLITS = 5
CHANCE_LEVEL = 'phase_rand'  # phase_rand ,shuffle or none
ITERATIONS = 1000  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# %%
# Get list of all recordings that have histology
sessions = query_sessions()

# decoding_result = pd.DataFrame()
# for i in range(len(sessions)):
for i in range(58, len(sessions)):
    print('Processing session %d of %d' % (i+1, len(sessions)))

    # Load in data
    eid = sessions[i]['url'][-36:]
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        ses_path = one.path_from_eid(eid)
        trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
    except:
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue
    if type(spikes) == tuple:
        continue

    # Get trial vectors
    incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] == 0.8).astype(int)

    # Check for number of trials
    if trial_times.shape[0] < MIN_TRIALS:
        continue

    # Decode per brain region
    for p, probe in enumerate(spikes.keys()):

        # Check if histology is available for this probe
        if not hasattr(clusters[probe], 'acronym'):
            continue

        # Get brain regions and combine cortical layers
        regions = combine_layers_cortex(np.unique(clusters[probe]['acronym']))

        # Decode per brain region
        for r, region in enumerate(np.unique(regions)):

            # Get clusters in this brain region
            region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
            clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]

            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                         clusters_in_region)]

            # Exclude neurons that drift over the session
            if EXCL_DRIFT_NEURONS:
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

            # Decode block identity
            decode_result = decode(spks_region, clus_region, trial_times, trial_blocks,
                                   pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                   cross_validation=VALIDATION, num_splits=NUM_SPLITS)

            # Estimate chance level
            if CHANCE_LEVEL == 'phase_rand':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_blocks,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       phase_rand=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'shuffle':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_blocks,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       shuffle=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'none':
                decode_chance = []
            else:
                raise Exception('CHANCE_LEVEL must be phase_rand, shuffle or none')

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': sessions[i]['subject'],
                                 'date': sessions[i]['start_time'][:10],
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'f1': decode_result['f1'].mean(),
                                 'accuracy': decode_result['accuracy'].mean(),
                                 'auroc': decode_result['auroc'].mean(),
                                 'chance_accuracy': [np.array(decode_chance['accuracy'])],
                                 'chance_f1': [np.array(decode_chance['f1'])],
                                 'chance_auroc': [np.array(decode_chance['auroc'])]}))

    if EXCL_DRIFT_NEURONS:
        decoding_result.to_pickle(join(SAVE_PATH,
                                       ('decoding_block_regions_no_drift_neurons_%s.p' % DECODER)))
    else:
        decoding_result.to_pickle(join(SAVE_PATH,
                                       ('decoding_block_regions_all_neurons_%s.p' % DECODER)))

