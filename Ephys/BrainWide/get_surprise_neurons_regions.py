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
from sklearn.utils import shuffle
from ephys_functions import paths, figure_style
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
rec_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track')
recordings = pd.DataFrame(data={
                            'eid': [rec['session']['id'] for rec in rec_with_hist],
                            'probe': [rec['probe_name'] for rec in rec_with_hist],
                            'date': [rec['session']['start_time'][:10] for rec in rec_with_hist],
                            'subject': [rec['session']['subject'] for rec in rec_with_hist]})

# Get list of eids of ephysChoiceWorld sessions
eids = one.search(dataset_types=['spikes.times', 'probes.trajectory'],
                  task_protocol='_iblrig_tasks_ephysChoiceWorld')

# Select only the ephysChoiceWorld sessions and sort by eid
recordings = recordings[recordings['eid'].isin(eids)]
recordings = recordings.sort_values('eid').reset_index()

surprise_neurons = pd.DataFrame()
for i, eid in enumerate(recordings['eid'].values):

    # Load in data (only when not already loaded from other probe)
    print('Processing recording %d of %d' % (i+1, len(recordings)))
    if i == 0:
        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
            trials = one.load_object(eid, 'trials')
        except:
            continue
    elif recordings.loc[i-1, 'eid'] != recordings.loc[i, 'eid']:
        try:
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
            trials = one.load_object(eid, 'trials')
        except:
            continue

    # Get probe
    probe = recordings.loc[i, 'probe']
    if probe not in spikes.keys():
        continue

    # Check data integrity
    if ((not hasattr(trials, 'stimOn_times'))
            or (len(trials.feedback_times) != len(trials.feedbackType))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        continue

    # Get trial vectors
    incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

    # Decode per brain region
    for j, region in enumerate(np.unique(clusters[probe]['acronym'])):

        # Get clusters in this brain region with KS2 label 'good'
        clusters_in_region = clusters[probe].metrics.cluster_id[
            (clusters[probe]['acronym'] == region) & (clusters[probe].metrics.ks2_label == 'good')]

        # Check if there are enough neurons in this brain region
        if np.shape(clusters_in_region)[0] < MIN_NEURONS:
            continue

        # Select spikes and clusters
        spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
        clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]

        # Get trial indices
        r_in_l_block = trials.stimOn_times[((trials.probabilityLeft > 0.55)
                                            & (trials.contrastRight > 0.1))]
        r_in_r_block = trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                            & (trials.contrastRight > 0.1))]
        l_in_r_block = trials.stimOn_times[((trials.probabilityLeft < 0.45)
                                            & (trials.contrastLeft > 0.1))]
        l_in_l_block = trials.stimOn_times[((trials.probabilityLeft > 0.55)
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
                                index=[0], data={'subject': recordings.loc[i, 'subject'],
                                                 'date': recordings.loc[i, 'date'],
                                                 'eid': recordings.loc[i, 'eid'],
                                                 'probe': probe,
                                                 'region': region,
                                                 'n_neurons': len(np.unique(clus_region)),
                                                 'n_sig_surprise': sig_units.shape[0]}))
    surprise_neurons.to_csv(join(SAVE_PATH, 'n_surprise_neurons_regions.csv'))

# %% Plot
# surprise_neurons = pd.read_csv(join(SAVE_PATH, 'n_surprise_neurons_regions.csv'))

surprise_summed = surprise_neurons.groupby('region').sum()
surprise_summed = surprise_summed.reset_index()
surprise_summed = surprise_summed[surprise_summed['n_neurons'] > 100]
surprise_summed['perc'] = (surprise_summed['n_sig_surprise'] / surprise_summed['n_neurons']) * 100
surprise_summed = surprise_summed.sort_values('perc', ascending=False)

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))

sns.barplot(x='perc', y='region', data=surprise_summed)
ax1.set(xlabel='Surprise neurons (%)', ylabel='')
figure_style(font_scale=1.1)
