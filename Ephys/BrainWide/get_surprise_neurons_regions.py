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
from ephys_functions import paths, figure_style, check_trials, sessions_with_hist
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
ALPHA = 0.05
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
        trials = one.load_object(eid, 'trials')
    except:
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue
    if type(spikes) != dict:
        continue

    # Get trial vectors
    incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

    # Decode per brain region
    for p, probe in enumerate(spikes.keys()):

        # Decode per brain region
        for j, region in enumerate(np.unique(clusters[probe]['acronym'])):
    
            # Get clusters in this brain region 
            clusters_in_region = clusters[probe].metrics.cluster_id[
                                                            clusters[probe]['acronym'] == region]
    
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
                                                & (trials.contrastRight > 0.1))]
            r_in_r_block = trials.stimOn_times[((trials.probabilityLeft == 0.2)
                                                & (trials.contrastRight > 0.1))]
            l_in_r_block = trials.stimOn_times[((trials.probabilityLeft == 0.2)
                                                & (trials.contrastLeft > 0.1))]
            l_in_l_block = trials.stimOn_times[((trials.probabilityLeft == 0.8)
                                                & (trials.contrastLeft > 0.1))]
    
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
        surprise_neurons.to_csv(join(SAVE_PATH, 'n_surprise_neurons_regions.csv'))

# %% Plot
surprise_neurons = pd.read_csv(join(SAVE_PATH, 'n_surprise_neurons_regions.csv'))
surprise_neurons = surprise_neurons[surprise_neurons['n_neurons'] > 50]
surprise_neurons['percentage'] = (surprise_neurons['n_sig_surprise']
                                  / surprise_neurons['n_neurons']) * 100
surprise_neurons['region'] = surprise_neurons['region'].astype(str)


surprise_per_region = surprise_neurons.groupby('region').mean()
surprise_per_region['n_neurons']  = surprise_neurons.groupby('region').sum()['n_neurons']
surprise_per_region = surprise_per_region.reset_index()
surprise_per_region = surprise_per_region.sort_values('percentage', ascending=False)
surprise_per_region = surprise_per_region[surprise_per_region['n_neurons'] > 100]

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))

sns.barplot(x='percentage', y='region', data=surprise_neurons)
ax1.set(xlabel='Surprise neurons (%)', ylabel='')
figure_style(font_scale=1.1)
