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
from brainbox.population import regress, get_spike_counts_in_bins
import pandas as pd
from scipy.stats import wilcoxon
from matplotlib.patches import Rectangle
import seaborn as sns
import alf
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from behavior_models import utils
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from my_functions import paths, combine_layers_cortex, figure_style, load_trials, remap
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

TARGET = 'prior-prevaction'
PRE_TIME = 0.6
POST_TIME = -0.1
DECODER = 'linear-regression'
ATLAS = 'beryl-atlas'
NUM_SPLITS = 5
VALIDATION = 'kfold'
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Ephys', 'Decoding', 'Sessions', DECODER, VALIDATION)
INCL_NEURONS = 'all'
INCL_SESSIONS = 'aligned-behavior'
CHANCE_LEVEL = 'pseudo'
N_SESSIONS = 10
PLOT_TRIALS = 400
BEFORE = 5
AFTER = 20
COLORS = (sns.color_palette('colorblind', as_cmap=True)[0],
          sns.color_palette('colorblind', as_cmap=True)[3])

# %%
decoding_result = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
       ('%s_%s_%s_%s_%s_cells_%s.p' % (TARGET, CHANCE_LEVEL, VALIDATION,
                                       INCL_SESSIONS, INCL_NEURONS, ATLAS))))

# Get decoding performance over chance
decoding_result['r_over_chance'] = decoding_result['r_prior'] - decoding_result['r_prior_null']
decoding_result = decoding_result.sort_values(by='r_prior', ascending=False).reset_index(drop=True)

for i in range(N_SESSIONS):

    # Load in data
    eid = decoding_result.loc[i, 'eid']
    probe = decoding_result.loc[i, 'probe']
    region = decoding_result.loc[i, 'region']
    subject = decoding_result.loc[i, 'subject']
    all_eids = np.sort(decoding_result.loc[decoding_result['subject'] == subject, 'eid'].unique())
    print('Processing %d of %d (region %s)' % (i+1, N_SESSIONS, region))

    # Load in fitted behavioral model
    if TARGET == 'prior-prevaction':
        model = exp_prev_action(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                                all_eids, subject,
                                np.array([np.array(None)] * all_eids.shape[0]),
                                np.array([np.array(None)] * all_eids.shape[0]),
                                np.array([np.array(None)] * all_eids.shape[0]))
    elif TARGET == 'prior-stimside':
        model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                             all_eids, subject,
                             np.array([np.array(None)] * all_eids.shape[0]),
                             np.array([np.array(None)] * all_eids.shape[0]),
                             np.array([np.array(None)] * all_eids.shape[0]))
    try:
        model.load_or_train(nb_steps=2000, remove_old=False)
    except:
        continue
    params = model.get_parameters(parameter_type='posterior_mean')

    # Get priors per trial
    trials = load_trials(eid)
    stim_side, stimuli, actions, prob_left = utils.format_data(trials)
    priors = model.compute_prior(np.array(actions), np.array(stimuli), np.array(stim_side),
                                 parameter_type='posterior_mean')[0]

    # Get clusters in this brain region
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)

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

    # Get clusters in this brain region
    clusters_in_region = [x for x, y in enumerate(clusters_regions) if (region == y) and (x in clusters_pass)]

    # Select spikes and clusters
    spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
    clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                 clusters_in_region)]


    # Decode prior from model fit
    times = np.column_stack(((trials.goCue_times - PRE_TIME),
                             (trials.goCue_times + POST_TIME)))
    population_activity, cluster_ids = get_spike_counts_in_bins(spks_region,
                                                                clus_region, times)
    population_activity = population_activity.T
    if VALIDATION == 'kfold-interleaved':
        cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    elif VALIDATION == 'kfold':
        cv = KFold(n_splits=NUM_SPLITS, shuffle=False)
    pred_prior = regress(population_activity, priors, cross_validation=cv)
    r_prior = pearsonr(priors, pred_prior)[0]

    # Plot trial-to-trial probability
    figure_style(font_scale=1.5)
    trial_blocks = (trials.probabilityLeft == 0.2).astype(int)
    f, ax1 = plt.subplots(1, 1, figsize=(12, 6), dpi=150)

    block_trans = np.append([0], np.array(np.where(np.diff(trials.probabilityLeft) != 0)) + 1)
    block_trans = np.append(block_trans, [trial_blocks.shape[0]])
    for j, trans in enumerate(block_trans[:-1]):
        if j == 0:
            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.1, alpha=0.5,
                          color=[.5, .5, .5])
        else:
            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.1, alpha=0.5,
                          color=COLORS[trial_blocks[trans]])
        ax1.add_patch(p)
    ax1.plot(priors, color='k', lw=1.5, label='Model output')
    ax1.set(xlim=[90, trial_blocks.shape[0]], ylim=[-0.05, 1.05],
            ylabel='Prior', xlabel='Trials',
            title='tau: %.2f' % (1/params[0]),
            yticks=[0, 1], yticklabels=['R', 'L'])
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_model' % (region, decoding_result.loc[i, 'subject'],
                                                      decoding_result.loc[i, 'date'],
                                                      decoding_result.loc[i, 'probe'])))
    plt.close(f)

    # Plot trial-to-trial decoding
    figure_style(font_scale=2)
    trial_blocks = (trials.probabilityLeft == 0.2).astype(int)
    f, ax1 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)

    block_trans = np.append([0], np.array(np.where(np.diff(trials.probabilityLeft) != 0)) + 1)
    block_trans = np.append(block_trans, [trial_blocks.shape[0]])
    for j, trans in enumerate(block_trans[:-1]):
        if j == 0:
            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.1, alpha=0.5,
                          color=[.5, .5, .5])
        else:
            p = Rectangle((trans, -0.05), block_trans[j+1] - trans, 1.1, alpha=0.5,
                          color=COLORS[trial_blocks[trans]])
        ax1.add_patch(p)
    ax1.plot(priors, color='k', lw=1.5, label='Model output')
    ax1.plot(pred_prior, color='r', lw=1.5, label='Decoding prediction')
    ax1.set(xlim=[0, PLOT_TRIALS], ylim=[-0.05, 1.05],
            ylabel='Prior', xlabel='Trials',
            title='Region %s; decoding performance: %.1f r' % (region, r_prior),
            yticks=[0, 1], yticklabels=['R', 'L'])
    #ax1.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_trials' % (region, decoding_result.loc[i, 'subject'],
                                                       decoding_result.loc[i, 'date'],
                                                       decoding_result.loc[i, 'probe'])))
    plt.close(f)

    # Plot around change points
    figure_style(font_scale=1.5)
    change_points = pd.DataFrame()
    for t, change_ind in enumerate(block_trans[2:-1]):
        if trials.probabilityLeft[change_ind] == 0.8:
            change_to = 'L'
        else:
            change_to = 'R'
        if change_ind < pred_prior.shape[0] - AFTER:
            change_points = change_points.append(pd.DataFrame(data={
                'pred_prior': pred_prior[change_ind-BEFORE:change_ind+AFTER],
                'trial': np.concatenate((np.arange(-BEFORE, 0), np.arange(1, AFTER+1))),
                'change_to': change_to}),
                ignore_index=True)

    f, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    sns.lineplot(data=change_points, x='trial', y='pred_prior', hue='change_to', ci=68,
                 palette=COLORS, hue_order=['L', 'R'], ax=ax1)
    legend = ax1.legend(frameon=False)
    ax1.plot([0, 0], [-0.5, 1], ls='--', color=[0.6, 0.6, 0.6])
    ax1.set(xlabel='Trials relative to block switch', ylabel='Predicted prior',
            yticks=[0, 1], yticklabels=['R', 'L'], ylim=[-0.1, 1])
    plt.tight_layout()
    sns.despine(trim=True)
    plt.savefig(join(FIG_PATH, '%s_%s_%s_%s_switches' % (region, decoding_result.loc[i, 'subject'],
                                                          decoding_result.loc[i, 'date'],
                                                          decoding_result.loc[i, 'probe'])))
    plt.close(f)

