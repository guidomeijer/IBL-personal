#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:21:49 2021

@author: guido
"""

import pandas as pd
import numpy as np
from os.path import join
import seaborn as sns
import matplotlib.pyplot as plt
from my_functions import (paths, query_sessions, check_trials, combine_layers_cortex, load_trials,
                          remap)
from brainbox.population import get_spike_counts_in_bins
import brainbox.io.one as bbone
import statsmodels.api as sm
from statsmodels.formula.api import ols
from ibllib.atlas import BrainRegions
from oneibl.one import ONE
one = ONE()
br = BrainRegions()
fig_path = join(paths()[1], '5HT', 'switch_neurons_brainwide')
save_path = join(paths()[2], '5HT')

# Settings
INCL_NEURONS = 'pass-QC'
INCL_SESSIONS = 'aligned-behavior'
ATLAS = 'beryl-atlas'
MIN_NEURONS = 5
MIN_CONTRAST = 0.1
PRE_TRIALS = 2
POST_TRIALS = 5
PRE_TIME = 0
POST_TIME = 0.3

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

    # Process trials
    trials = trials.loc[trials['probabilityLeft'] != 0.5]  # Exclude 50/50 block
    left_trials = trials[(trials['stim_side'] == -1) & (trials['signed_contrast'].abs() > MIN_CONTRAST)].reset_index()
    left_trans = np.array(np.where(np.diff(left_trials['probabilityLeft']) > 0.5)[0]) + 1
    right_trials = trials[(trials['stim_side'] == 1) & (trials['signed_contrast'].abs() > MIN_CONTRAST)].reset_index()
    right_trans = np.array(np.where(np.diff(right_trials['probabilityLeft']) < -0.5)[0]) + 1

    # Decode per brain region
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
            mapped_br = br.get(ids=remap(clusters[probe]['atlas_id']))
            clusters_regions = mapped_br['acronym']
        elif ATLAS == 'allen-atlas':
            clusters_regions = combine_layers_cortex(clusters[probe]['acronym'])

        # Get list of neurons that pass QC
        if INCL_NEURONS == 'pass-QC':
            clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]
        elif INCL_NEURONS == 'all':
            clusters_pass = np.arange(clusters[probe]['metrics'].shape[0])

        # Decode per brain region
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

            # Create dataframe
            for t, trans in enumerate(left_trans):
                if trans < left_trials.shape[0] - POST_TRIALS:
                    stim_times = left_trials.loc[trans-PRE_TRIALS:trans+(POST_TRIALS-1), 'stimOn_times'].values
                    times = np.column_stack((stim_times - PRE_TIME, stim_times + POST_TIME))
                    population_activity, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
                    df = pd.DataFrame(data=population_activity,
                                      columns=np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS) + 1))
                    df['neuron_id'] = cluster_ids
                    df = df.melt(id_vars=['neuron_id'], var_name='trial', value_name='spike_count')
                    df['trans_to'] = 'L'
                    region_df = region_df.append(df)
            for t, trans in enumerate(right_trans):
                if trans < right_trials.shape[0] - POST_TRIALS:
                    stim_times = right_trials.loc[trans-PRE_TRIALS:trans+(POST_TRIALS-1), 'stimOn_times'].values
                    times = np.column_stack((stim_times - PRE_TIME, stim_times + POST_TIME))
                    population_activity, cluster_ids = get_spike_counts_in_bins(spks_region, clus_region, times)
                    df = pd.DataFrame(data=population_activity,
                                      columns=np.append(np.arange(-PRE_TRIALS, 0), np.arange(0, POST_TRIALS) + 1))
                    df['neuron_id'] = cluster_ids
                    df = df.melt(id_vars=['neuron_id'], var_name='trial', value_name='spike_count')
                    df['trans_to'] = 'R'
                    region_df = region_df.append(df)

            # Calculate spike rate
            region_df['spike_rate'] = region_df['spike_count'] * (1 / (PRE_TIME + POST_TIME))

            # Calculate significance
            p_values = np.zeros(cluster_ids.shape[0])
            for n, neuron in enumerate(region_df['neuron_id'].unique()):
                mod = ols('spike_rate ~ trial', data=region_df[region_df['neuron_id'] == neuron]).fit()
                aov_table = sm.stats.anova_lm(mod, typ=2)
                p_values[n] = aov_table.loc['trial', 'PR(>F)']

                if p_values[n] < 0.05:
                    # Plot significant neurons
                    f, ax1 = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
                    sns.lineplot(x='trial', y='spike_rate', ax=ax1, err_style='bars', ci=68,
                                 data=region_df[region_df['neuron_id'] == neuron])
                    ax1.set(xticks=np.append(np.arange(-PRE_TRIALS,0), np.arange(0,POST_TRIALS)+1),
                            xlabel='Trials relative to block switch', ylabel='Spike rate (spks/s)',
                            title='%s' % region)
                    sns.despine(trim=True)
                    plt.savefig(join(fig_path, '%s_%s_%s_%s' % (region, neuron, subject, date)))
                    plt.close(f)

            # Add to dataframe
            results_df.loc[results_df.shape[0] + 1, 'percentage'] = (np.sum(p_values < 0.05)
                                                                     / p_values.shape[0] * 100)
            results_df.loc[results_df.shape[0], 'n_neurons'] = p_values.shape[0]
            results_df.loc[results_df.shape[0], 'region'] = region
            results_df.loc[results_df.shape[0], 'subject'] = subject
            results_df.loc[results_df.shape[0], 'date'] = date
            results_df.loc[results_df.shape[0], 'eid'] = eid
            results_df.loc[results_df.shape[0], 'probe'] = probe

    # Save intermediate results
    results_df.to_csv(join(save_path, 'switch_neurons.csv'))



