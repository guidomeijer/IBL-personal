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
import math
import seaborn as sns
from sklearn.utils import shuffle
from scipy.stats import pearsonr
from ephys_functions import paths, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
SAMPLING_RATE = 30000
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

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

    # Decode per brain region
    for j, region in enumerate(np.unique(clusters[probe]['acronym'])):

        # Get clusters in this brain region with KS2 label 'good'
        clusters_in_region = clusters[probe].metrics.cluster_id[
            (clusters[probe]['acronym'] == region) & (clusters[probe].metrics.ks2_label == 'good')]

        # Select spikes and clusters
        spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
        clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, clusters_in_region)]

        # Calculate time constant per neuron
        time_constant = []
        for k, cluster in enumerate(np.unique(clus_region)):



            # The time constant is defined as the time it takes for the autocorrelation function
            # to decay by a factor of e
            auto_corr = acf(spks_region[clus_region == cluster], nlags=30000, fft=False)
            tc = (np.argmin(np.abs(auto_corr - 1 / math.e)) / SAMPLING_RATE) * 1000
            time_constant.append(tc)

        # Add to dataframe
        decoding_result = decoding_result.append(pd.DataFrame(
            index=[0], data={'subject': recordings.loc[i, 'subject'],
                             'date': recordings.loc[i, 'date'],
                             'eid': recordings.loc[i, 'eid'],
                             'probe': probe,
                             'region': region,
                             'time_constant': time_constant}))

    decoding_result.to_csv(join(SAVE_PATH, 'time_constant_regions'))

# %% Plot
decoding_result = pd.read_csv(join(SAVE_PATH, 'time_constant_regions'))

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
