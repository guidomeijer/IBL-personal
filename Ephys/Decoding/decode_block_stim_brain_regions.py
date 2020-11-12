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
POST_TIME = 0.2
MIN_NEURONS = 5  # min neurons per region
STIM_SIDE = 'right'
DECODER = 'bayes'
VALIDATION = 'leave-one-out'
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'behavior_crit'
CHANCE_LEVEL = 'phase_rand'  # phase_rand, shuffle or none
ITERATIONS = 1000  # for null distribution estimation
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# %%
# Get list of all recordings that have histology
sessions = query_sessions(selection=INCL_SESSIONS)

# Detect data format
if 'model' in sessions[0]:
    ses_type = 'insertions'
elif 'good_enough_for_brainwide_map' in sessions[0]:
    ses_type = 'datajoint'
else:
    ses_type = 'sessions'

decoding_result = pd.DataFrame()
for i in range(len(sessions)):
    print('Processing session %d of %d' % (i+1, len(sessions)))

    # Extract eid based on data format
    if ses_type == 'insertions':
        eid = sessions[i]['session']
    elif ses_type == 'datajoint':
        eid = str(sessions[i]['session_uuid'])
    elif ses_type == 'sessions':
        eid = sessions[i]['url'][-36:]

    # Load in data
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True,
                                                                           one=one)
        ses_path = one.path_from_eid(eid)
        trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
    except Exception as error_message:
        print(error_message)
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Get trial vectors
    if STIM_SIDE == 'right':
        incl_trials = ((~np.isnan(trials.contrastRight) | (trials.contrastLeft == 0))
                       & (trials.probabilityLeft == 0.8))
        incl_trials[np.random.choice(np.where(((~np.isnan(trials.contrastRight)
                                    | (trials.contrastLeft == 0))
                                   & (trials.probabilityLeft == 0.2)))[0],
                         size=np.sum(incl_trials), replace=False)] = True
    elif STIM_SIDE == 'left':
        incl_trials = ((~np.isnan(trials.contrastLeft) | (trials.contrastRight == 0))
                       & (trials.probabilityLeft == 0.2))
        incl_trials[np.random.choice(np.where(((~np.isnan(trials.contrastLeft)
                                    | (trials.contrastRight == 0))
                                   & (trials.probabilityLeft == 0.8)))[0],
                         size=np.sum(incl_trials), replace=False)] = True
    trial_times = trials.stimOn_times[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

    # Extract session data depending on data format
    if ses_type == 'insertions':
        subject = sessions[i]['session_info']['subject']
        date = sessions[i]['session_info']['start_time'][:10]
        probes_to_use = [sessions[i]['name']]
    elif ses_type == 'datajoint':
        subject = sessions[i]['subject_nickname']
        date = str(sessions[i]['session_ts'].date())
        probes_to_use = spikes.keys()
    else:
        subject = sessions[i]['subject']
        date = sessions[i]['start_time'][:10]
        probes_to_use = spikes.keys()

    # Calculate bias shift for this subject
    blank_left = trials.choice[((trials.contrastLeft == 0) | (trials.contrastRight == 0))
                               & (trials.probabilityLeft == 0.8) & incl_trials]
    blank_right = trials.choice[((trials.contrastLeft == 0) | (trials.contrastRight == 0))
                                & (trials.probabilityLeft == 0.2) & incl_trials]
    prop_right_l = np.sum(blank_left == 1) / np.sum((blank_left == 1) | (blank_left == -1))
    prop_right_r = np.sum(blank_right == 1) / np.sum((blank_right == 1) | (blank_right == -1))
    bias = prop_right_l - prop_right_r

    # Decode per brain region
    for p, probe in enumerate(probes_to_use):

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

            # Decode block identity
            decode_result = decode(spks_region, clus_region, trial_times, trial_blocks,
                                   pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                   cross_validation=VALIDATION)

            # Estimate chance level
            if CHANCE_LEVEL == 'phase_rand':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_blocks,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                       cross_validation=VALIDATION,
                                       phase_rand=True, iterations=ITERATIONS)
            elif CHANCE_LEVEL == 'shuffle':
                decode_chance = decode(spks_region, clus_region, trial_times, trial_blocks,
                                       pre_time=PRE_TIME, post_time=POST_TIME, classifier=DECODER,
                                       cross_validation=VALIDATION,
                                       shuffle=True, iterations=ITERATIONS)
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
                                 'n_trials': trial_times.shape[0],
                                 'n_neurons': np.unique(clus_region).shape[0]}))

    decoding_result.to_pickle(join(SAVE_PATH,
       ('decode_block_stim_%s_%s_neurons_%s_sessions.p' % (DECODER, INCL_NEURONS, INCL_SESSIONS))))
