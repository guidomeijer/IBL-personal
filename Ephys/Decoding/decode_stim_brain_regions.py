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
PRE_TIME = 0
POST_TIME = 0.3
MIN_NEURONS = 5  # min neurons per region
MIN_TRIALS = 300
DECODER = 'bayes'
VALIDATION = 'kfold'
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'all'  # all or aligned
MIN_CONTRAST = 0.1
NUM_SPLITS = 5
CHANCE_LEVEL = 'phase_rand'  # phase_rand, shuffle or none
ITERATIONS = 1000  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')
DOWNLOAD_TRIALS = False

# %%
# Get list of all recordings that have histology
sessions = query_sessions(selection=INCL_SESSIONS)

# Detect if is a list of sessions or insertions
if 'model' in sessions[0]:
    ses_type = 'insertions'
else:
    ses_type = 'sessions'

decoding_result = pd.DataFrame()
for i in range(len(sessions)):
    print('Processing session %d of %d' % (i+1, len(sessions)))

    # Detect if sessions is a list of sessions or insertions
    if 'model' in sessions[i]:
        eid = sessions[i]['session']
    else:
        eid = sessions[i]['url'][-36:]

    # Load in data
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        ses_path = one.path_from_eid(eid)
        if DOWNLOAD_TRIALS:
            _ = one.load(eid, dataset_types=['trials.stimOn_times', 'trials.probabilityLeft'],
                         download_only=True, clobber=True)
        trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
    except:
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue
    if type(spikes) == tuple:
        continue

    # Get trial vectors
    incl_trials = (trials.contrastLeft > MIN_CONTRAST) | (trials.contrastRight > MIN_CONTRAST)
    trial_times = trials.stimOn_times[incl_trials]
    trial_ids = np.isnan(trials.contrastLeft[incl_trials]).astype(int)

    # Check for number of trials
    if trial_times.shape[0] < MIN_TRIALS:
        continue

    # Extract session data depending on whether input is a list of sessions or insertions
    if 'model' in sessions[i]:
        subject = sessions[i]['session_info']['subject']
        date = sessions[i]['session_info']['start_time'][:10]
        probes_to_use = [sessions[i]['name']]
    else:
        subject = sessions[i]['subject']
        date = sessions[i]['start_time'][:10]
        probes_to_use = spikes.keys()

    # Decode per brain region
    for p, probe in enumerate(probes_to_use):

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

            # Check if there are enough neurons in this brain region
            if np.unique(clus_region).shape[0] < MIN_NEURONS:
                continue

            # Decode block identity
            decode_result = decode(spks_region, clus_region, trial_times, trial_ids,
                                   pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                   cross_validation=VALIDATION, num_splits=NUM_SPLITS)

            # Estimate chance level
            if CHANCE_LEVEL == 'phase_rand':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_ids,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       phase_rand=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'shuffle':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_ids,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                       cross_validation=VALIDATION, num_splits=NUM_SPLITS,
                                       shuffle=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'none':
                decode_chance = []
            else:
                raise Exception('CHANCE_LEVEL must be phase_rand, shuffle or none')

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': subject,
                                 'date': date,
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'f1': decode_result['f1'].mean(),
                                 'accuracy': decode_result['accuracy'].mean(),
                                 'auroc': decode_result['auroc'].mean(),
                                 'chance_accuracy': [np.array(decode_chance['accuracy'])],
                                 'chance_f1': [np.array(decode_chance['f1'])],
                                 'chance_auroc': [np.array(decode_chance['auroc'])],
                                 'n_trials': trial_ids.shape[0],
                                 'n_neurons': np.unique(clus_region).shape[0]}))

    decoding_result.to_pickle(join(SAVE_PATH,
       ('decode_stim_side_%s_%s_neurons_%s_sessions.p' % (DECODER, INCL_NEURONS, INCL_SESSIONS))))
