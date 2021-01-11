#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import scipy
from brainbox.population import decode, _get_spike_counts_in_bins
import pandas as pd
from scipy.stats import wilcoxon
from matplotlib.patches import Rectangle
import seaborn as sns
import alf
from ephys_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'bayes-multinomial'
ITERATIONS = 1000
VALIDATION = 'kfold-interleaved'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Sessions', DECODER, 'block')
INCL_NEURONS = 15 # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
CHANCE_LEVEL = 'pseudo-session'
N_SESSIONS = 10
METRIC = 'accuracy'
BEFORE = 5
AFTER = 20
REGION = 'IF'
COLORS = (sns.color_palette('colorblind', as_cmap=True)[0],
          sns.color_palette('colorblind', as_cmap=True)[3])

# %%
decoding_result = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % ('block', CHANCE_LEVEL, VALIDATION,
                                    INCL_SESSIONS, INCL_NEURONS))))

# Exclude root
decoding_result = decoding_result.reset_index(drop=True)
incl_regions = [i for i, j in enumerate(decoding_result['region']) if not j.islower()]
decoding_result = decoding_result.loc[incl_regions]

# Drop duplicates
decoding_result = decoding_result[decoding_result.duplicated(subset=['region', 'eid', 'probe'])
                                  == False]

# Get decoding performance over chance
decoding_result['%s_over_chance' % METRIC] = (
            decoding_result[METRIC] - decoding_result['chance_%s' % METRIC])
decoding_result = decoding_result.sort_values(by='%s_over_chance' % METRIC,
                                              ascending=False).reset_index(drop=True)

# for i in decoding_result[decoding_result['region'] == REGION].index:
for i in range(N_SESSIONS):

    # Load in data
    eid = decoding_result.loc[i, 'eid']
    probe = decoding_result.loc[i, 'probe']
    region = decoding_result.loc[i, 'region']
    print('Processing %d of %d (region %s)' % (i+1, N_SESSIONS, region))
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)
    ses_path = one.path_from_eid(eid)
    trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

    # Get trial vectors
    incl_trials = (trials.probabilityLeft == 0.8) | (trials.probabilityLeft == 0.2)
    trial_times = trials.stimOn_times[incl_trials]
    probability_left = trials.probabilityLeft[incl_trials]
    trial_blocks = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

    # Get clusters in this brain region
    region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
    clusters_in_region = clusters[probe].metrics.cluster_id[region_clusters == region]

    # Select spikes and clusters
    spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
    clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                 clusters_in_region)]

    if len(spks_region) == 0:
        continue

    # Decode block identity
    decode_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                          pre_time=PRE_TIME, post_time=POST_TIME,
                          classifier=DECODER, cross_validation=VALIDATION,
                          num_splits=5)

    pseudo_block = decode(spks_region, clus_region, trial_times, trial_blocks,
                          pre_time=PRE_TIME, post_time=POST_TIME,
                          classifier=DECODER, cross_validation=VALIDATION,
                          num_splits=5, pseudo_blocks=True, iterations=ITERATIONS)

    # Plot trial-to-trial probability
    figure_style(font_scale=1.5)
    trial_blocks = (trials.probabilityLeft == 0.2).astype(int)
    f, ax1 = plt.subplots(1, 1, figsize=(12, 6), dpi=150)

    block_trans = np.append([0], np.array(np.where(np.diff(trials.probabilityLeft) != 0)) + 1)
    block_trans = np.append(block_trans, [trial_blocks.shape[0]])
    for j, trans in enumerate(block_trans[:-1]):
        p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.1, alpha=0.5,
                      color=COLORS[trial_blocks[trans]])
        ax1.add_patch(p)
    ax1.plot(np.arange(90, trial_blocks.shape[0]),
             np.convolve(decode_block['probabilities'][0], np.ones(5), 'same') / 5,
             lw=1.5, color=[0.4, 0.4, 0.4])
    ax1.plot(np.arange(90, trial_blocks.shape[0]), decode_block['probabilities'][0],
             'o', lw=2, color='k')
    ax1.set(xlim=[90, trial_blocks.shape[0]], ylim=[-0.05, 1.05],
            ylabel='Block classification probability', xlabel='Trials',
            title='Region %s; decoding accuracy over chance: %.1f%%' % (
                region, (decode_block['accuracy'] - pseudo_block['accuracy'].mean()) * 100),
            yticks=[0, 1], yticklabels=['L', 'R'])
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_trials' % (region, decoding_result.loc[i, 'subject'],
                                                       decoding_result.loc[i, 'date'],
                                                       decoding_result.loc[i, 'probe'])))
    plt.close(f)

    # Plot probability around change points
    figure_style(font_scale=1.5)
    change_points = pd.DataFrame()
    for t, change_ind in enumerate(block_trans[2:-1]):
        if trials.probabilityLeft[change_ind] == 0.8:
            change_to = 'L'
        else:
            change_to = 'R'
        if change_ind - 90 < decode_block['probabilities'][0].shape[0] - AFTER:
            change_points = change_points.append(pd.DataFrame(data={
                'probability': decode_block['probabilities'][0][
                                                (change_ind-90)-BEFORE:(change_ind-90)+AFTER],
                'trial': np.concatenate((np.arange(-BEFORE, 0), np.arange(1, AFTER+1))),
                'change_to': change_to}),
                ignore_index=True)

    f, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    sns.lineplot(data=change_points, x='trial', y='probability', hue='change_to', ci=68,
                 palette=COLORS, ax=ax1)
    legend = ax1.legend(frameon=False)
    ax1.plot([0, 0], [-0.5, 1], ls='--', color=[0.6, 0.6, 0.6])
    ax1.set(xlabel='Trials relative to block switch', ylabel='Classification probability',
            yticks=[0, 1], yticklabels=['L', 'R'], ylim=[-0.1, 1])
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_switches' % (region, decoding_result.loc[i, 'subject'],
                                                          decoding_result.loc[i, 'date'],
                                                          decoding_result.loc[i, 'probe'])))
    plt.close(f)

    # Plot trial-to-trial probability
    figure_style(font_scale=1.5)
    trial_blocks = (trials.probabilityLeft == 0.2).astype(int)
    f, ax1 = plt.subplots(1, 1, figsize=(12, 6), dpi=150)

    for j, trans in enumerate(block_trans[:-1]):
        p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.1, alpha=0.5,
                      color=COLORS[trial_blocks[trans]])
        ax1.add_patch(p)
    ax1.plot(np.arange(90, trial_blocks.shape[0]),
             np.convolve(pseudo_block['probabilities'][0], np.ones(5), 'same') / 5,
             lw=1.5, color=[0.4, 0.4, 0.4])
    ax1.plot(np.arange(90, trial_blocks.shape[0]), pseudo_block['probabilities'][0],
             'o', lw=2, color='k')
    ax1.set(xlim=[90, trial_blocks.shape[0]], ylim=[-0.05, 1.05],
            ylabel='Block classification probability', xlabel='Trials',
            title='Region %s; decoding accuracy over chance: %.1f%%' % (
                region, (decode_block['accuracy'] - pseudo_block['accuracy'].mean()) * 100),
            yticks=[0, 1], yticklabels=['L', 'R'])
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_trials_pseudo' % (
                    region, decoding_result.loc[i, 'subject'], decoding_result.loc[i, 'date'],
                    decoding_result.loc[i, 'probe'])))
    plt.close(f)

    # Plot probability around change points
    figure_style(font_scale=1.5)
    change_points = pd.DataFrame()
    for t, change_ind in enumerate(block_trans[2:-1]):
        if trials.probabilityLeft[change_ind] == 0.8:
            change_to = 'L'
        else:
            change_to = 'R'
        if change_ind - 90 < pseudo_block['probabilities'][0].shape[0] - AFTER:
            change_points = change_points.append(pd.DataFrame(data={
                'probability': pseudo_block['probabilities'][0][
                                                (change_ind-90)-BEFORE:(change_ind-90)+AFTER],
                'trial': np.concatenate((np.arange(-BEFORE, 0), np.arange(1, AFTER+1))),
                'change_to': change_to}),
                ignore_index=True)

    f, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    sns.lineplot(data=change_points, x='trial', y='probability', hue='change_to', ci=68,
                 palette=COLORS, ax=ax1)
    legend = ax1.legend(frameon=False)
    ax1.plot([0, 0], [-0.5, 1], ls='--', color=[0.6, 0.6, 0.6])
    ax1.set(xlabel='Trials relative to block switch', ylabel='Classification probability',
            yticks=[0, 1], yticklabels=['L', 'R'], ylim=[-0.1, 1])
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_switches_pseudo' % (region,
                                                                decoding_result.loc[i, 'subject'],
                                                          decoding_result.loc[i, 'date'],
                                                          decoding_result.loc[i, 'probe'])))
    plt.close(f)

