#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from brainbox.population import _get_spike_counts_in_bins
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
import alf
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import KFold
from sklearn.utils import shuffle as sklearn_shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from my_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

PRE_TIME = 0
POST_TIME = 0.3
DECODER = 'bayes-multinomial'
ITERATIONS = 1000
VALIDATION = 'kfold-interleaved'
NUM_SPLITS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'Sessions', DECODER, 'block-stim')
INCL_NEURONS = 'all'  # all or no_drift
INCL_SESSIONS = 'aligned-behavior'  # all or aligned
CHANCE_LEVEL = 'shuffle'
N_SESSIONS = 20
METRIC = 'accuracy'
BEFORE = 5
AFTER = 20
COLORS = (sns.color_palette('colorblind', as_cmap=True)[0],
          sns.color_palette('colorblind', as_cmap=True)[3])

# %%
decoding_result = pd.read_pickle(join(SAVE_PATH, DECODER,
       ('%s_%s_%s_%s_%s_cells.p' % ('block-stim', CHANCE_LEVEL, VALIDATION,
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

# Initialize decoder
if DECODER == 'bayes-multinomial':
    clf = MultinomialNB()
else:
    clf = GaussianNB()


def decode(pop_vector, trial_ids, num_splits, interleaved):

    # Cross-validation
    if interleaved:
        cv = KFold(n_splits=num_splits, shuffle=True).split(pop_vector)
    else:
        cv = KFold(n_splits=num_splits).split(pop_vector)

    # Loop over the splits into train and test
    y_pred = np.zeros(trial_ids.shape)
    y_probs = np.zeros(trial_ids.shape)
    for train_index, test_index in cv:

        # Fit the model to the training data
        clf.fit(pop_vector[train_index], trial_ids[train_index])

        # Predict the test data
        y_pred[test_index] = clf.predict(pop_vector[test_index])

        # Get the probability of the prediction for ROC analysis
        probs = clf.predict_proba(pop_vector[test_index])
        y_probs[test_index] = probs[:, 1]  # keep positive only

    return y_pred, y_probs


def balanced_trial_set(trials):
    # A balanced random subset of left and right trials from the two blocks
    left_stim = (trials.contrastLeft > 0) & (trials.probabilityLeft == 0.2)
    left_stim[np.random.choice(np.where(((trials.contrastLeft > 0)
                                         & (trials.probabilityLeft == 0.8)))[0],
                               size=np.sum(left_stim), replace=False)] = True
    right_stim = (trials.contrastRight > 0) & (trials.probabilityLeft == 0.8)
    right_stim[np.random.choice(np.where(((trials.contrastRight > 0)
                                         & (trials.probabilityLeft == 0.2)))[0],
                               size=np.sum(right_stim), replace=False)] = True
    incl_trials = (left_stim | right_stim
                   | (((trials.contrastLeft == 0) | (trials.contrastRight == 0))
                      & (trials.probabilityLeft != 0.5)))
    return incl_trials


# %%
for i in range(N_SESSIONS):
    print('Processing %d of %d' % (i+1, N_SESSIONS))

    # Load in data
    eid = decoding_result.loc[i, 'eid']
    probe = decoding_result.loc[i, 'probe']
    region = decoding_result.loc[i, 'region']
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)
    ses_path = one.path_from_eid(eid)
    trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

    # Get clusters in this brain region
    region_clusters = combine_layers_cortex(clusters[probe]['acronym'])
    clusters_in_region = clusters[probe].metrics.cluster_id[clusters[probe]['acronym'] == region]

    # Select spikes and clusters
    spks_region = spikes[probe].times[np.isin(spikes[probe].clusters, clusters_in_region)]
    clus_region = spikes[probe].clusters[np.isin(spikes[probe].clusters,
                                                 clusters_in_region)]

    # Get matrix of all neuronal responses
    times = np.column_stack(((trials.stimOn_times - PRE_TIME),
                             (trials.stimOn_times + POST_TIME)))
    pop_vector, cluster_ids = _get_spike_counts_in_bins(spks_region, clus_region, times)
    pop_vector = pop_vector.T

    # Subselect trials to balance stimulus sides
    accuracy = np.empty(ITERATIONS)
    accuracy_shuffle = np.empty(ITERATIONS)
    for k in range(ITERATIONS):

        # A balanced random subset of left and right trials from the two blocks
        incl_trials = balanced_trial_set(trials)

        # Select activity matrix and trial ids for this iteration
        this_pop_vector = pop_vector[incl_trials]
        trial_ids = (trials.probabilityLeft[incl_trials] == 0.2).astype(int)

        if k == 0:
            pred = np.empty((ITERATIONS, this_pop_vector.shape[0]))
            prob = np.empty((ITERATIONS, this_pop_vector.shape[0]))
            pred_shuffle = np.empty((ITERATIONS, this_pop_vector.shape[0]))
            prob_shuffle = np.empty((ITERATIONS, this_pop_vector.shape[0]))
            trial_numbers = np.empty((ITERATIONS, np.sum(incl_trials)))

        # Get trial numbers of this iteration
        trial_numbers[k, :] = np.where(incl_trials)[0]

        # Decode
        if VALIDATION[6:] == 'interleaved':
            y_pred, y_probs = decode(this_pop_vector, trial_ids, NUM_SPLITS, True)
        else:
            y_pred, y_probs = decode(this_pop_vector, trial_ids, NUM_SPLITS, False)
        accuracy[k] = accuracy_score(trial_ids, y_pred)
        pred[k, :] = y_pred
        prob[k, :] = y_probs

        # Decode shuffled data
        if VALIDATION[6:] == 'interleaved':
            y_pred, y_probs = decode(this_pop_vector, sklearn_shuffle(trial_ids),
                                     NUM_SPLITS, True)
        else:
            y_pred, y_probs = decode(this_pop_vector, sklearn_shuffle(trial_ids),
                                     NUM_SPLITS, False)
        accuracy_shuffle[k] = accuracy_score(trial_ids, y_pred)
        pred_shuffle[k, :] = y_pred
        prob_shuffle[k, :] = y_probs

    # Get average probability per trial
    unique_trial_numbers = np.unique(trial_numbers)
    prob_per_trial = np.empty(unique_trial_numbers.shape[0])
    for t, trial in enumerate(unique_trial_numbers):
        prob_per_trial[t] = np.mean(prob[np.where(trial_numbers == trial)])

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
    ax1.plot(unique_trial_numbers, np.convolve(prob_per_trial, np.ones(5), 'same') / 5,
             lw=1.5, color=[0.4, 0.4, 0.4])
    ax1.plot(unique_trial_numbers, prob_per_trial, 'o', lw=2, color='k')

    ax1.set(xlim=[90, trial_blocks.shape[0]], ylim=[-0.05, 1.05],
            ylabel='Block classification probability', xlabel='Trials',
            title='Region %s; decoding accuracy over chance: %.1f%%' % (
                                    region, (np.mean(accuracy) - np.mean(accuracy_shuffle)) * 100),
            yticks=[0, 1], yticklabels=['L', 'R'])
    plt.tight_layout()
    plt.savefig(join(FIG_PATH, '%s_%s_%s_trials' % (region, decoding_result.loc[i, 'subject'],
                                                    decoding_result.loc[i, 'date'])))
    plt.close(f)

    # Plot probability around change points
    figure_style(font_scale=1.5)
    change_points = pd.DataFrame()
    for t, change in enumerate(block_trans[2:-1]):
        change_ind = np.where(unique_trial_numbers == change)[0][0]
        if trials.probabilityLeft[change] == 0.8:
            change_to = 'L'
        else:
            change_to = 'R'
        if change_ind < prob_per_trial.shape[0] - AFTER:
            change_points = change_points.append(pd.DataFrame(data={
                'probability': prob_per_trial[change_ind - BEFORE : change_ind + AFTER],
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
    plt.savefig(join(FIG_PATH, '%s_%s_%s_switches' % (region, decoding_result.loc[i, 'subject'],
                                                      decoding_result.loc[i, 'date'])))
    plt.close(f)

