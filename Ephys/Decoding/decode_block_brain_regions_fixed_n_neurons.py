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
import pandas as pd
import alf
from ephys_functions import (paths, sessions_with_hist, check_trials,
                             combine_layers_cortex)
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
PRE_TIME = 0.6
POST_TIME = -0.1
MIN_NEURONS = 15  # min neurons per region
N_NEURONS = 15  # number of neurons to use for decoding
MIN_TRIALS = 300
ITERATIONS = 1000
DECODER = 'bayes'
VALIDATION = 'kfold'
NUM_SPLITS = 5
COMBINE_LAYERS_CORTEX = True
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# %%
# Get list of all recordings that have histology
ses_with_hist = sessions_with_hist()

decoding_result = pd.DataFrame()
for i in range(len(ses_with_hist)):
    print('Processing session %d of %d' % (i+1, len(ses_with_hist)))

    # Load in data
    eid = ses_with_hist[i]['url'][-36:]
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
        for r, region in enumerate(regions):

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
            decode_result = decode(spks_region, clus_region,
                                   trial_times, trial_blocks,
                                   pre_time=PRE_TIME, post_time=POST_TIME,
                                   classifier=DECODER, cross_validation=VALIDATION,
                                   num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                                   iterations=ITERATIONS)
            """
            # Shuffle
            shuffle_result = decode(spks_region, clus_region,
                                    trial_times, trial_blocks,
                                    pre_time=PRE_TIME, post_time=POST_TIME,
                                    classifier=DECODER, cross_validation=VALIDATION,
                                    num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                                    iterations=ITERATIONS, shuffle=True)
            """

            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': ses_with_hist[i]['subject'],
                                 'date': ses_with_hist[i]['start_time'][:10],
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'f1': decode_result['f1'].mean(),
                                 'accuracy': decode_result['accuracy'].mean(),
                                 'auroc': decode_result['auroc'].mean()}))

    decoding_result.to_csv(join(SAVE_PATH,
                                ('decoding_block_regions_%d_neurons_%s.csv'
                                 % (N_NEURONS, DECODER))))
