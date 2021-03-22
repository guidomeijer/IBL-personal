#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:21:49 2021

@author: guido
"""

import pandas as pd
import numpy as np
from os.path import join, isdir
import seaborn as sns
from os import mkdir
import matplotlib.pyplot as plt
from brainbox.lfp import butter_filter
from matplotlib.patches import Rectangle
from my_functions import (paths, query_sessions, check_trials, combine_layers_cortex, load_trials,
                          remap)
from brainbox.task import differentiate_units
from brainbox.population import get_spike_counts_in_bins
import brainbox.io.one as bbone
from scipy.stats import pearsonr
from oneibl.one import ONE
one = ONE()
fig_path = join(paths()[1], 'Ephys', 'SingleNeurons', 'block_neurons')
save_path = join(paths()[2], 'Ephys')

# Settings
INCL_NEURONS = 'pass-QC'
INCL_SESSIONS = 'aligned-behavior'
ATLAS = 'beryl-atlas'
MIN_NEURONS = 1
PLOT = True
PRE_TIME = 0.6
POST_TIME = -0.1
TRIAL_CENTERS = np.arange(-10, 21, 3)
TRIAL_WIN = 5
BASELINE_TRIAL_WIN = 5
COLORS = (sns.color_palette('colorblind', as_cmap=True)[0],
          sns.color_palette('colorblind', as_cmap=True)[3])

# Query session list
eids, probes = query_sessions(selection=INCL_SESSIONS)

results_df = pd.DataFrame()
for i in range(len(eids)):
    print('\nProcessing session %d of %d' % (i+1, len(eids)))

    # Load in data
    eid = eids[i]
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
                                                                    eid, aligned=True, one=one)
        ses_path = one.path_from_eid(eid)
        trials = load_trials(eid)
    except Exception as error_message:
        print(error_message)
        continue

    # Check data integrity
    if check_trials(trials) is False:
        continue

    # Extract session data
    ses_info = one.get_details(eid)
    subject = ses_info['subject']
    date = ses_info['start_time'][:10]
    probes_to_use = probes[i]

    # Process per probe
    for p, probe in enumerate(probes_to_use):
        print('Processing %s (%d of %d)' % (probe, p + 1, len(probes_to_use)))

        # Check if data is available for this probe
        if probe not in clusters.keys():
            continue

        # Check if histology is available for this probe
        if not hasattr(clusters[probe], 'acronym'):
            continue

        # Check if cluster metrics are available
        if 'metrics' not in clusters[probe]:
            continue

        # Get list of brain regions
        if ATLAS == 'beryl-atlas':
            clusters_regions = remap(clusters[probe]['atlas_id'])
        elif ATLAS == 'allen-atlas':
            clusters_regions = combine_layers_cortex(clusters[probe]['acronym'])

        # Get list of neurons that pass QC
        if INCL_NEURONS == 'pass-QC':
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        elif INCL_NEURONS == 'all':
            clusters_pass = np.arange(clusters[probe]['metrics'].shape[0])

        # Process per brain region
        for r, region in enumerate(np.unique(clusters_regions)):

            # Skip region if any of these conditions apply
            if region.islower():
                continue

            print('Processing region %s (%d of %d)' % (region, r + 1, len(np.unique(clusters_regions))))
            region_df = pd.DataFrame()

            # Get clusters in this brain region
            clusters_in_region = [x for x, y in enumerate(clusters_regions)
                                  if (region == y) and (x in clusters_pass)]

            # Select spikes and clusters
            spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
            clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                         clusters_in_region)]

            # Check if there are enough neurons in this brain region
            if np.unique(clus_region).shape[0] < MIN_NEURONS:
                continue

            # Get population activity per trials
            times = np.column_stack((trials['stimOn_times'] - PRE_TIME, trials['stimOn_times'] + POST_TIME))
            population_activity, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
            population_activity = population_activity * (1 / (PRE_TIME + POST_TIME))  # spks/s

            # Determine significant neurons
            stim_times = trials['stimOn_times'][trials['probabilityLeft'] != 0.5]
            pleft = trials['probabilityLeft'][trials['probabilityLeft'] != 0.5]
            pleft = (pleft == 0.8).astype(int)
            sig_block, _, p_values, _ = differentiate_units(spks_region, clus_region, stim_times, pleft,
                                                            pre_time=PRE_TIME, post_time=POST_TIME)

            # Add to dataframe
            results_df.loc[results_df.shape[0] + 1, 'percentage'] = (np.sum(p_values < 0.05)
                                                                     / p_values.shape[0] * 100)
            results_df.loc[results_df.shape[0], 'n_sig_neurons'] = np.sum(p_values < 0.05)
            results_df.loc[results_df.shape[0], 'n_neurons'] = p_values.shape[0]
            results_df.loc[results_df.shape[0], 'region'] = region
            results_df.loc[results_df.shape[0], 'subject'] = subject
            results_df.loc[results_df.shape[0], 'date'] = date
            results_df.loc[results_df.shape[0], 'eid'] = eid
            results_df.loc[results_df.shape[0], 'probe'] = probe

            if PLOT:
                # Subtract baseline
                baseline = np.empty(population_activity.shape)
                for n in range(population_activity.shape[0]):
                    fit = np.poly1d(np.polyfit(np.arange(population_activity.shape[1]),
                                               population_activity[n, :], 2))
                    baseline[n, :] = fit(np.arange(population_activity.shape[1]))
                pop_baseline = population_activity - baseline

                # Get activity around block switches
                switch_to_l = [i for i, x in enumerate(np.diff(trials.probabilityLeft) > 0.3) if x]
                switch_to_r = [i for i, x in enumerate(np.diff(trials.probabilityLeft) < -0.3) if x]
                all_switches = np.append(switch_to_l, switch_to_r)
                switch_sides = np.append(['left']*len(switch_to_l), ['right']*len(switch_to_r))
                block_switch = pd.DataFrame(columns=['mean_spike_count', 'trial_center',
                                                     'switch_side', 'cluster_id'])
                for s, switch in enumerate(all_switches):
                    for t, trial in enumerate(TRIAL_CENTERS):
                        this_counts = pop_baseline[np.isin(cluster_ids, sig_block),
                                                   int(switch+(trial-(TRIAL_WIN/2))):int(
                                                       switch+(trial+(TRIAL_WIN/2)))]
                        block_switch = block_switch.append(pd.DataFrame(
                                                data={'spike_count': np.mean(this_counts, axis=1),
                                                      'trial_center': trial,
                                                      'cluster_id': sig_block,
                                                      'switch_side': switch_sides[s]}), sort=False)

                # Plot sigificant neurons
                for n, neuron_id in enumerate(sig_block):

                    # Apply some smoothing
                    spike_rate = np.convolve(np.squeeze(population_activity[cluster_ids == neuron_id, :]),
                                             np.ones((5,))/5, mode='same')

                    # Plot
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
                    block_trans = np.append([0], np.array(np.where(np.diff(trials.probabilityLeft) != 0)) + 1)
                    block_trans = np.append(block_trans, [len(trials)])
                    for j, trans in enumerate(block_trans[:-1]):
                        if j == 0:
                            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, spike_rate.max(), alpha=0.5,
                                          color=[.5, .5, .5])
                        else:
                            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, spike_rate.max(), alpha=0.5,
                                          color=COLORS[pleft[trans]])
                        ax1.add_patch(p)
                    ax1.plot(np.arange(1, len(trials) + 1), spike_rate, color='k')
                    ax1.set(xlabel='Trials', ylabel='Firing rate (spks/s)', title='Region: %s' % region)

                    sns.lineplot(x='trial_center', y='spike_count', hue='switch_side',
                                 data=block_switch.loc[block_switch['cluster_id'] == neuron_id], ci=68,
                                 palette=COLORS, hue_order=['right', 'left'], lw=2)
                    y_lim = ax2.get_ylim()
                    ax2.plot([0, 0], y_lim, color=[0.6, 0.6, 0.6], linestyle='dashed')
                    ax2.set(ylabel='Baseline subtracted spike rate (spk/s)',
                           xlabel='Trials relative to block switch')
                    legend = ax2.legend(bbox_to_anchor=(1, 0.9), frameon=False)
                    #legend.texts[0].set_text('Block switch to')
                    plt.tight_layout()
                    sns.set(context='paper', font_scale=1.5, style='ticks')
                    sns.despine(trim=True)
                    if not isdir(join(fig_path, region)):
                        mkdir(join(fig_path, region))
                    plt.savefig(join(fig_path, region, '%s_%s_%s_cluster-%s' % (
                        subject, date, probe, cluster_ids[n])))
                    plt.close(f)

                    # Plot baseline subtraction
                    f, ax1 = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
                    ax1.plot(np.arange(1, len(trials) + 1),
                             np.squeeze(population_activity[cluster_ids == neuron_id, :]),
                             color='k', label='spike rate')
                    ax1.plot(np.arange(1, len(trials) + 1),
                             np.squeeze(baseline[cluster_ids == neuron_id, :]),
                             color='r', lw=2, label='baseline')
                    ax1.plot(np.arange(1, len(trials) + 1),
                             np.squeeze(pop_baseline[cluster_ids == neuron_id, :]),
                             color='g', label='subtracted')
                    ax1.set(xlabel='Trials', ylabel='Firing rate (spks/s)', title='Region: %s' % region)
                    ax1.legend(frameon=False)

                    plt.tight_layout()
                    sns.set(context='paper', font_scale=1.5, style='ticks')
                    sns.despine()

                    plt.savefig(join(fig_path, region, '%s_%s_%s_cluster-%s_baseline' % (
                        subject, date, probe, cluster_ids[n])))
                    plt.close(f)

    # Save intermediate results
    results_df.to_csv(join(save_path, 'block_neurons.csv'))