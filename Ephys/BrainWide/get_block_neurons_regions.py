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
from brainbox.population import decode
import pandas as pd
import seaborn as sns
import alf
from sklearn.utils import shuffle
from ephys_functions import (paths, figure_style, check_trials, sessions_with_hist,
                             combine_layers_cortex)
import brainbox as bb
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 0.6
POST_TIME = -0.1
MIN_NEURONS = 1
ALPHA = 0.05
COMBINE_LAYERS_CORTEX = True
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# %%
# Get list of all recordings that have histology
session_list = sessions_with_hist()

block_neurons = pd.DataFrame()
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
    incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] == 0.8).astype(int)

    for p, probe in enumerate(spikes.keys()):

        # Check if histology is available
        if not hasattr(clusters[probe], 'acronym'):
            continue       
        
        # Get brain regions and combine cortical layers
        if COMBINE_LAYERS_CORTEX:
            regions = combine_layers_cortex(np.unique(clusters[probe]['acronym']))
        else:
            regions = np.unique(clusters[probe]['acronym'])
                
        for i, region in enumerate(regions):
            
            # Get clusters in this brain region 
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
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]
    
            # Get block neurons
            sig_block = bb.task.differentiate_units(spks_region, clus_region,
                                                    trial_times, trial_blocks,
                                                    pre_time=PRE_TIME, post_time=POST_TIME,
                                                    alpha=ALPHA)[0]
    
            # Add to dataframe
            block_neurons = block_neurons.append(pd.DataFrame(
                                    index=[0], data={'subject': session_list[i]['subject'],
                                                     'date': session_list[i]['start_time'][:10],
                                                     'eid': eid,
                                                     'probe': probe,
                                                     'region': region,
                                                     'n_neurons': len(np.unique(clus_region)),
                                                     'n_sig_block': sig_block.shape[0]}))
            
    if COMBINE_LAYERS_CORTEX:
        block_neurons.to_csv(join(SAVE_PATH, 'n_block_neurons_combined_regions.csv'))
    else:
        block_neurons.to_csv(join(SAVE_PATH, 'n_block_neurons_regions.csv'))

