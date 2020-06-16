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
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 0.6
POST_TIME = -0.1
MIN_NEURONS = 20  # min neurons per region
N_NEURONS = 20  # number of neurons to use for decoding
MIN_TRIALS = 300
ITERATIONS = 1000
DECODER = 'bayes'  # bayes, regression or forest
VALIDATION = 'kfold'
NUM_SPLITS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

# %%
# Get list of all recordings that have histology
rec_with_hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                              task_protocol='_iblrig_tasks_ephysChoiceWorld')

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

decoding_result = pd.DataFrame()
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

    # Check for number of trials
    if trial_times.shape[0] < MIN_TRIALS:
        continue

    # Decode per brain region
    for i, region in enumerate(np.unique(clusters[probe]['acronym'])):

        # Get clusters in this brain region with KS2 label 'good'
        clusters_in_region = clusters[probe].metrics.cluster_id[
            (clusters[probe]['acronym'] == region) & (clusters[probe].metrics.ks2_label == 'good')]

        # Check if there are enough neurons in this brain region
        if np.shape(clusters_in_region)[0] < MIN_NEURONS:
            continue

        # Select spikes and clusters
        spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
        clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]

        # Decode block identity
        decode_result = decode(spks_region, clus_region,
                               trial_times, trial_blocks,
                               pre_time=PRE_TIME, post_time=POST_TIME,
                               classifier=DECODER, cross_validation=VALIDATION,
                               num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                               iterations=ITERATIONS)

        # Shuffle
        shuffle_result = decode(spikes[probe].times, spikes[probe].clusters,
                                trial_times, shuffle(trial_blocks),
                                pre_time=PRE_TIME, post_time=POST_TIME,
                                classifier=DECODER, cross_validation=VALIDATION,
                                num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                                iterations=ITERATIONS, shuffle=True)

        # Add to dataframe
        decoding_result = decoding_result.append(pd.DataFrame(
            index=[0], data={'subject': recordings.loc[i, 'subject'],
                             'date': recordings.loc[i, 'date'],
                             'eid': recordings.loc[i, 'eid'],
                             'probe': probe,
                             'region': region,
                             'f1': decode_result['f1'].mean() - shuffle_result['f1'].mean(),
                             'accuracy': (decode_result['accuracy'].mean()
                                          - shuffle_result['accuracy'].mean()),
                             'auroc': (decode_result['auroc'].mean()
                                       - shuffle_result['auroc'].mean())}))

    decoding_result.to_csv(join(SAVE_PATH, 'decoding_block_all_regions_%d_neurons' % N_NEURONS))

# %% Plot
decoding_result = pd.read_csv(join(SAVE_PATH, 'decoding_block_all_regions_20_neurons'))

decoding_regions = decoding_result.groupby('region').size()[
                                    decoding_result.groupby('region').size() > 3].reset_index()
decoding_result = decoding_result[decoding_result['region'].isin(decoding_regions['region'])]
decoding_result = decoding_result.sort_values('accuracy', ascending=False)

sort_regions = decoding_result.groupby('region').mean().sort_values(
                                            'accuracy', ascending=False).reset_index()['region']

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))

sns.barplot(x='accuracy', y='region', data=decoding_result, order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding accuracy of stimulus prior (% over chance)', ylabel='')
figure_style(font_scale=1.2)