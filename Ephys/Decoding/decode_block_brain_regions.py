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
from scipy.stats import wilcoxon
import seaborn as sns
import alf
from ephys_functions import paths, figure_style, sessions_with_hist, check_trials
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

# Settings
DOWNLOAD = False
OVERWRITE = False
PRE_TIME = 0.6
POST_TIME = -0.1
MIN_NEURONS = 15  # min neurons per region
N_NEURONS = 15  # number of neurons to use for decoding
MIN_TRIALS = 300
ITERATIONS = 1000
DECODER = 'bayes'  # bayes, regression or forest
VALIDATION = 'kfold'
NUM_SPLITS = 5
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

        # Decode per brain region
        for r, region in enumerate(np.unique(clusters[probe]['acronym'])):
    
            # Get clusters in this brain region with KS2 label 'good'
            clusters_in_region = clusters[probe].metrics.cluster_id[
                                                    (clusters[probe]['acronym'] == region)]
    
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
    
            # Shuffle
            shuffle_result = decode(spikes[probe].times, spikes[probe].clusters,
                                    trial_times, trial_blocks,
                                    pre_time=PRE_TIME, post_time=POST_TIME,
                                    classifier=DECODER, cross_validation=VALIDATION,
                                    num_splits=NUM_SPLITS, n_neurons=N_NEURONS,
                                    iterations=ITERATIONS, shuffle=True)
    
            # Add to dataframe
            decoding_result = decoding_result.append(pd.DataFrame(
                index=[0], data={'subject': ses_with_hist[i]['subject'],
                                 'date': ses_with_hist[i]['start_time'][:10],
                                 'eid': eid,
                                 'probe': probe,
                                 'region': region,
                                 'f1': decode_result['f1'].mean(),
                                 'f1_shuffle': shuffle_result['f1'].mean(),
                                 'accuracy': decode_result['accuracy'].mean(),
                                 'accuracy_shuffle': shuffle_result['accuracy'].mean(),
                                 'auroc': decode_result['auroc'].mean(),
                                 'auroc_shuffle': shuffle_result['auroc'].mean()}))
    
        decoding_result.to_csv(join(SAVE_PATH,
                                    'decoding_block_all_regions_%d_neurons' % N_NEURONS))

# %% Plot
decoding_result = pd.read_csv(join(SAVE_PATH, 'decoding_block_all_regions_%d_neurons' % N_NEURONS))

p_value = 1
min_perf = 0.15
max_fano = 0.85
metric = 'f1'  # f1 or acc    

# Exclude root
decoding_result = decoding_result.reset_index()
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Get decoding performance over chance
decoding_result['acc_over_chance'] = (decoding_result['accuracy']
                                      - decoding_result['accuracy_shuffle'])
decoding_result['f1_over_chance'] = (decoding_result['f1'] - decoding_result['f1_shuffle'])

# Calculate significance and average decoding performance per region
for i, region in enumerate(decoding_result['region'].unique()):
    _, p = wilcoxon(decoding_result.loc[decoding_result['region'] == region, 'accuracy'],
                    decoding_result.loc[decoding_result['region'] == region,
                                        'accuracy_shuffle'])
    decoding_result.loc[decoding_result['region'] == region, 'p_value_acc'] = p
    _, p = wilcoxon(decoding_result.loc[decoding_result['region'] == region, 'f1'],
                    decoding_result.loc[decoding_result['region'] == region,
                                        'f1_shuffle'])
    decoding_result.loc[decoding_result['region'] == region, 'p_value_f1'] = p
    decoding_result.loc[decoding_result['region'] == region, 'acc_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'acc_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'f1_mean'] = decoding_result.loc[
                            decoding_result['region'] == region, 'f1_over_chance'].mean()
    decoding_result.loc[decoding_result['region'] == region, 'f1_fano'] = (
        decoding_result.loc[decoding_result['region'] == region, 'f1_over_chance'].std()
        / decoding_result.loc[decoding_result['region'] == region, 'f1_over_chance'].mean())
    
# Apply plotting thresholds
"""
decoding_result = decoding_result[(decoding_result['%s_mean' % metric] > min_perf)
                                  & (decoding_result['p_value_%s' % metric] < p_value)
                                  & (decoding_result['%s_fano' % metric] < max_fano)]
"""
decoding_result = decoding_result[(decoding_result['%s_mean' % metric] > min_perf)]

# Get sorting
sort_regions = decoding_result.groupby('region').mean().sort_values(
                            '%s_over_chance' % metric, ascending=False).reset_index()['region']

f, ax1 = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(x='%s_over_chance' % metric, y='region', data=decoding_result,
            order=sort_regions, ci=68, ax=ax1)
ax1.set(xlabel='Decoding accuracy of stimulus prior (f1 score over chance)', ylabel='')
figure_style(font_scale=1.2)
plt.savefig(join(FIG_PATH, 'decode_block_regions_%d_neurons' % N_NEURONS))
