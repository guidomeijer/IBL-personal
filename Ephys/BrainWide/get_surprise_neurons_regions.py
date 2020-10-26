#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:56:57 2020

Decode left/right block identity from all brain regions

@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import alf
from ephys_functions import (paths, figure_style, check_trials, sessions_with_hist,
                             combine_layers_cortex)
import brainbox as bb
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 0
POST_TIME = 0.5
MIN_NEURONS = 1
MIN_CONTRAST = 0.1
ALPHA = 0.05
COMBINE_LAYERS_CORTEX = True
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# %%
# Get list of all recordings that have histology
session_list = sessions_with_hist()

surprise_neurons = pd.DataFrame()
for i in range(len(session_list)):
    print('Processing session %d of %d' % (i+1, len(session_list)))
    
    # Load in data
    eid = session_list[i]['url'][-36:]
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        ses_path = one.path_from_eid(eid)
        trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')
    except:
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Get trial vectors
    incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

    # Decode per brain region
    for p, probe in enumerate(spikes.keys()):
        
        # Check if histology is available for this probe
        if not hasattr(clusters[probe], 'acronym'):
            continue    
        
        # Get brain regions and combine cortical layers
        if COMBINE_LAYERS_CORTEX:
            regions = combine_layers_cortex(np.unique(clusters[probe]['acronym']))
        else:
            regions = np.unique(clusters[probe]['acronym'])

        # Decode per brain region
        for j, region in enumerate(regions):
    
            # Get brain regions and combine cortical layers
            if COMBINE_LAYERS_CORTEX:
                region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
            else:
                region_clusters = clusters[probe]['acronym']
    
            # Get clusters in this brain region 
            clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]
    
            # Check if there are enough neurons in this brain region
            if np.shape(clusters_in_region)[0] < MIN_NEURONS:
                continue
    
            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters,
                                                      clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                         clusters_in_region)]
    
            # Get trial indices
            r_in_l_block = trials.stimOn_times[((trials.probabilityLeft == 0.8)
                                                & (trials.contrastRight > MIN_CONTRAST))]
            r_in_r_block = trials.stimOn_times[((trials.probabilityLeft == 0.2)
                                                & (trials.contrastRight > MIN_CONTRAST))]
            l_in_r_block = trials.stimOn_times[((trials.probabilityLeft == 0.2)
                                                & (trials.contrastLeft > MIN_CONTRAST))]
            l_in_l_block = trials.stimOn_times[((trials.probabilityLeft == 0.8)
                                                & (trials.contrastLeft > MIN_CONTRAST))]
    
            # Get significant units
            r_units = bb.task.differentiate_units(spks_region, clus_region,
                                                  np.append(r_in_l_block,
                                                            r_in_r_block),
                                                  np.append(np.zeros(len(r_in_l_block)),
                                                            np.ones(len(r_in_r_block))),
                                                  pre_time=PRE_TIME, post_time=POST_TIME,
                                                  test='ranksums', alpha=0.05)[0]
            l_units = bb.task.differentiate_units(spks_region, clus_region,
                                                  np.append(l_in_l_block,
                                                            l_in_r_block),
                                                  np.append(np.zeros(len(l_in_l_block)),
                                                            np.ones(len(l_in_r_block))),
                                                  pre_time=PRE_TIME, post_time=POST_TIME,
                                                  test='ranksums', alpha=0.05)[0]
            sig_units = np.unique(np.concatenate((l_units, r_units)))
    
            # Add to dataframe
            surprise_neurons = surprise_neurons.append(pd.DataFrame(
                                    index=[0], data={'subject': session_list[i]['subject'],
                                                     'date': session_list[i]['start_time'][:10],
                                                     'eid': eid,
                                                     'probe': probe,
                                                     'region': region,
                                                     'n_neurons': len(np.unique(clus_region)),
                                                     'n_sig_surprise': sig_units.shape[0]}))
    if COMBINE_LAYERS_CORTEX:
        surprise_neurons.to_csv(join(SAVE_PATH, 'n_surprise_neurons_combined_regions.csv'))
    else:
        surprise_neurons.to_csv(join(SAVE_PATH, 'n_surprise_neurons_regions.csv'))

