#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:11:29 2020

@author: guido
"""

import seaborn as sns
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from my_functions import paths, check_trials
import brainbox.io.one as bbone
from brainbox.population import decode
from oneibl.one import ONE
one = ONE()

# Settings
REGION = 'ACAd6a'
WIN_CENTERS = np.arange(-1.2, 1.7, 0.18)
WIN_SIZE = 0.2
PLOT_X = [-1, 1.5]
DECODER = 'bayes'  # bayes, regression or forest
N_NEURONS = 10
NUM_SPLITS = 5
ITERATIONS = 1000
FIG_PATH = paths()[1]

# Query sessions with at least one channel in the region of interest
ses = one.alyx.rest('sessions', 'list', atlas_acronym=REGION,
                    task_protocol='_iblrig_tasks_ephysChoiceWorld',
                    project='ibl_neuropixel_brainwide')

# Loop over sessions
for i, eid in enumerate([j['url'][-36:] for j in ses]):
    print('Processing session %d of %d' % (i+1, len(ses)))

    # Load in data
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        trials = one.load_object(eid, 'trials')
    except:
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Get stim on times
    incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
    stimon_blocks = (trials.probabilityLeft[incl_trials] == 0.8).astype(int)
    stimon_times = trials.stimOn_times[incl_trials]

    # Loop over probes
    for p, probe in enumerate(spikes.keys()):

        # Get clusters in brain region of interest
        region_clusters = [ind for ind, s in enumerate(clusters[probe]['acronym']) if REGION in s]
        spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, region_clusters)]
        clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters, region_clusters)]
        if len(region_clusters) <= N_NEURONS:
            continue

        # Decode over time
        decode_time = pd.DataFrame()
        shuffle_time = pd.DataFrame()
        for j, win_center in enumerate(WIN_CENTERS):
            print('Decoding window [%d of %d]' % (j+1, WIN_CENTERS.shape[0]))
            decode_result = decode(spks_region, clus_region, stimon_times, stimon_blocks,
                                   pre_time=-win_center+(WIN_SIZE/2),
                                   post_time=win_center+(WIN_SIZE/2),
                                   classifier='bayes', cross_validation='kfold',
                                   n_neurons=N_NEURONS, iterations=ITERATIONS)
            decode_time = decode_time.append(pd.DataFrame({
                        'f1': decode_result['f1'], 'accuracy': decode_result['accuracy'] * 100,
                        'auroc': decode_result['auroc'], 'win_center': win_center,
                        'session': '%s_%s' % (ses[i]['subject'], ses[i]['start_time'][:10])}),
                        sort=False)
            shuffle_result = decode(spks_region, clus_region, stimon_times, stimon_blocks,
                                    pre_time=-win_center+(WIN_SIZE/2),
                                    post_time=win_center+(WIN_SIZE/2),
                                    classifier='bayes', cross_validation='kfold',
                                    n_neurons=N_NEURONS, iterations=ITERATIONS, shuffle=True)
            shuffle_time = shuffle_time.append(pd.DataFrame({
                        'f1': shuffle_result['f1'], 'accuracy': shuffle_result['accuracy'] * 100,
                        'auroc': shuffle_result['auroc'], 'win_center': win_center,
                        'session': '%s_%s' % (ses[i]['subject'], ses[i]['start_time'][:10])}),
                        sort=False)

# %% Plot

over_chance = pd.DataFrame()
over_chance['f1'] = decode_time['f1'] - shuffle_time['f1']
over_chance['accuracy'] = decode_time['accuracy'] - shuffle_time['accuracy']
over_chance['auroc'] = decode_time['auroc'] - shuffle_time['auroc']
over_chance['win_center'] = decode_time['win_center']

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
ax1.plot([0, 0], [-20, 20], linestyle='dashed', color=[0.6, 0.6, 0.6])
ax1.plot([-5, 5], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='accuracy', data=over_chance, ax=ax1, lw=2)
ax1.set(ylabel='Decoding performance\n(% correct over chance)', xlabel='Time (s)',
        title='Stimulus onset', xlim=PLOT_X)

ax2.plot([-5, 5], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
# ax2.plot([0, 0], [0, 100], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='f1', data=over_chance, ax=ax2, lw=2)
ax2.set(ylabel='Decoding performance\n(F1 score over chance)', xlabel='Time (s)',
        title='Stimulus onset', xlim=PLOT_X)

ax3.plot([-5, 5], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
# ax3.plot([0, 0], [0, 100], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='auroc', data=over_chance, ax=ax3, lw=2)
ax3.set(ylabel='Decoding performance\n(AUROC over chance)', xlabel='Time (s)',
        title='Stimulus onset', xlim=PLOT_X)

sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine()
plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, 'decoding_block_time_%s' % REGION))

