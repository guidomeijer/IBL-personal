# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:16:30 2020

Decode whether a stimulus is consistent or inconsistent with the block for frontal and control
recordings seperated by probe depth.

@author: guido
"""

from os import listdir
from os.path import join
import alf.io as ioalf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from brainbox.population import decode
from functions_5HT import paths
from sklearn.utils import shuffle

# Session
LAB = 'danlab'
SUBJECT = 'DY_011'
DATE = '2020-01-30'
PROBE = '00'

# Settings
WIN_CENTERS = np.arange(-1.2, 1.7, 0.18)
WIN_SIZE = 0.2
PLOT_X = [-1, 1.5]
DECODER = 'bayes'  # bayes, regression or forest
N_NEURONS = 150
NEURON_GROUPS = np.arange(10, 180, 10)
PRE_TIME = 0.6
POST_TIME = -0.1
NUM_SPLITS = 5
ITERATIONS = 1000

DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding')

# Get paths
ses_nr = listdir(join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE))[0]
session_path = join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE, ses_nr)
alf_path = join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE, ses_nr, 'alf')
probe_path = join(DATA_PATH, LAB, 'Subjects', SUBJECT, DATE, ses_nr, 'alf', 'probe%s' % PROBE)

# Load in data
spikes = ioalf.load_object(probe_path, 'spikes')
clusters = ioalf.load_object(probe_path, 'clusters')
trials = ioalf.load_object(alf_path, '_ibl_trials')

# Only use single units
spikes.times = spikes.times[np.isin(
        spikes.clusters, clusters.metrics.cluster_id[
                            clusters.metrics.ks2_label == 'good'])]
spikes.clusters = spikes.clusters[np.isin(
        spikes.clusters, clusters.metrics.cluster_id[
                            clusters.metrics.ks2_label == 'good'])]
clusters.channels = clusters.channels[clusters.metrics.ks2_label == 'good']
clusters.depths = clusters.depths[clusters.metrics.ks2_label == 'good']
cluster_ids = clusters.metrics.cluster_id[clusters.metrics.ks2_label == 'good']

# Get stim on times
incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
gocue_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)
gocue_times = trials.goCue_times[incl_trials]

"""

# Decode with increasing number of neurons
decode_groups = pd.DataFrame()
for i, group_size in enumerate(NEURON_GROUPS):
    print('Decoding group size %d [%d of %d]' % (group_size, i+1, NEURON_GROUPS.shape[0]))
    bayes_kfold = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                         pre_time=PRE_TIME, post_time=POST_TIME,
                         classifier='bayes', cross_validation='kfold', num_splits=NUM_SPLITS,
                         n_neurons=group_size, iterations=ITERATIONS)
    decode_groups = decode_groups.append(pd.DataFrame({'f1': bayes_kfold['f1'],
                                                       'auroc': bayes_kfold['auroc'],
                                                       'group_size': group_size}), sort=False)

"""

# Get reward times
reward_blocks = (trials.probabilityLeft[incl_trials
                                        & (trials.feedbackType == 1)] > 0.55).astype(int)
reward_times = trials.feedback_times[incl_trials & (trials.feedbackType == 1)]

# Get omission
omission_blocks = (trials.probabilityLeft[incl_trials
                                          & (trials.feedbackType == -1)] > 0.55).astype(int)
omission_times = trials.feedback_times[incl_trials & (trials.feedbackType == -1)]

# Decode over time
decode_time = pd.DataFrame()
shuffle_time = pd.DataFrame()
for i, win_center in enumerate(WIN_CENTERS):
    print('Decoding window [%d of %d]' % (i+1, WIN_CENTERS.shape[0]))
    # GO CUE
    decode_result = decode(spikes.times, spikes.clusters, gocue_times, gocue_blocks,
                           pre_time=-win_center+(WIN_SIZE/2), post_time=win_center+(WIN_SIZE/2),
                           classifier='bayes', cross_validation='kfold',
                           n_neurons=N_NEURONS, iterations=ITERATIONS)
    decode_time = decode_time.append(pd.DataFrame({'f1': decode_result['f1'],
                                                   'accuracy': decode_result['accuracy'] * 100,
                                                   'auroc': decode_result['auroc'],
                                                   'win_center': win_center,
                                                   'event': 'go cue'}), sort=False)
    shuffle_result = decode(spikes.times, spikes.clusters, gocue_times, shuffle(gocue_blocks),
                            pre_time=-win_center+(WIN_SIZE/2), post_time=win_center+(WIN_SIZE/2),
                            classifier='bayes', cross_validation='kfold',
                            n_neurons=N_NEURONS, iterations=ITERATIONS)
    shuffle_time = shuffle_time.append(pd.DataFrame({'f1': shuffle_result['f1'],
                                                     'accuracy': shuffle_result['accuracy'] * 100,
                                                     'auroc': shuffle_result['auroc'],
                                                     'win_center': win_center,
                                                     'event': 'go cue'}), sort=False)
    """
    # REWARD
    decode_result = decode(spikes.times, spikes.clusters,
                           reward_times, reward_blocks,
                           pre_time=-win_center+(WIN_SIZE/2), post_time=win_center+(WIN_SIZE/2),
                           classifier='bayes', cross_validation='kfold',
                           n_neurons=N_NEURONS, iterations=ITERATIONS)
    decode_time = decode_time.append(pd.DataFrame({'f1': decode_result['f1'],
                                                   'accuracy': decode_result['accuracy'] * 100,
                                                   'auroc': decode_result['auroc'],
                                                   'win_center': win_center,
                                                   'event': 'reward'}), sort=False)
    shuffle_result = decode(spikes.times, spikes.clusters,
                            reward_times, shuffle(reward_blocks),
                            pre_time=-win_center+(WIN_SIZE/2), post_time=win_center+(WIN_SIZE/2),
                            classifier='bayes', cross_validation='kfold',
                            n_neurons=N_NEURONS, iterations=ITERATIONS)
    shuffle_time = shuffle_time.append(pd.DataFrame({'f1': shuffle_result['f1'],
                                                     'accuracy': shuffle_result['accuracy'] * 100,
                                                     'auroc': shuffle_result['auroc'],
                                                     'win_center': win_center,
                                                     'event': 'reward'}), sort=False)

    # OMISSION
    decode_result = decode(spikes.times, spikes.clusters,
                           omission_times, omission_blocks,
                           pre_time=-win_center+(WIN_SIZE/2), post_time=win_center+(WIN_SIZE/2),
                           classifier='bayes', cross_validation='kfold',
                           n_neurons=N_NEURONS, iterations=ITERATIONS)
    decode_time = decode_time.append(pd.DataFrame({'f1': decode_result['f1'],
                                                   'accuracy': decode_result['accuracy'] * 100,
                                                   'auroc': decode_result['auroc'],
                                                   'win_center': win_center,
                                                   'event': 'omission'}), sort=False)
    shuffle_result = decode(spikes.times, spikes.clusters,
                            omission_times, shuffle(omission_blocks),
                            pre_time=-win_center+(WIN_SIZE/2), post_time=win_center+(WIN_SIZE/2),
                            classifier='bayes', cross_validation='kfold',
                            n_neurons=N_NEURONS, iterations=ITERATIONS)
    shuffle_time = shuffle_time.append(pd.DataFrame({'f1': shuffle_result['f1'],
                                                     'accuracy': shuffle_result['accuracy'] * 100,
                                                     'auroc': shuffle_result['auroc'],
                                                     'win_center': win_center,
                                                     'event': 'omission'}), sort=False)
    """

# %% Plot

over_chance = pd.DataFrame()
over_chance['f1'] = decode_time['f1'] - shuffle_time['f1']
over_chance['accuracy'] = decode_time['accuracy'] - shuffle_time['accuracy']
over_chance['auroc'] = decode_time['auroc'] - shuffle_time['auroc']
over_chance['event'] = decode_time['event']
over_chance['win_center'] = decode_time['win_center']

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
ax1.plot([0, 0], [-20, 20], linestyle='dashed', color=[0.6, 0.6, 0.6])
ax1.plot([-5, 5], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='accuracy', data=over_chance[over_chance['event'] == 'go cue'],
             ax=ax1, lw=2)
ax1.set(ylabel='Decoding performance\n(% correct over chance)', xlabel='Time (s)',
        title='Stimulus onset', xlim=PLOT_X, ylim=[-5, 12])

ax2.plot([-5, 5], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
# ax2.plot([0, 0], [0, 100], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='f1', data=over_chance[over_chance['event'] == 'go cue'],
             ax=ax2, lw=2)
ax2.set(ylabel='Decoding performance\n(F1 score over chance)', xlabel='Time (s)',
        title='Stimulus onset', xlim=PLOT_X)

ax3.plot([-5, 5], [0, 0], linestyle='dashed', color=[0.6, 0.6, 0.6])
# ax3.plot([0, 0], [0, 100], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='auroc', data=over_chance[over_chance['event'] == 'go cue'],
             ax=ax3, lw=2)
ax3.set(ylabel='Decoding performance\n(F1 score over chance)', xlabel='Time (s)',
        title='Stimulus onset', xlim=PLOT_X)

sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine()
plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, 'decoding_block_time_acc'))

"""
ax2.plot([0, 0], [0, 100], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='accuracy', data=shuffle_time[shuffle_time['event'] == 'reward'],
             ax=ax2, lw=2, color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='accuracy', data=decode_time[decode_time['event'] == 'reward'],
             ax=ax2, lw=2)
ax2.set(ylabel='Decoding performance (% correct)', xlabel='Time (s)', title='Reward delivery',
        ylim=[40, 60], xlim=PLOT_X)

chance_lvl = (omission_blocks.sum() / omission_blocks.shape[0]) * 100
ax3.plot([0, 0], [0, 100], linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='accuracy', data=shuffle_time[shuffle_time['event'] == 'omission'],
             ax=ax3, lw=2, color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='accuracy', data=decode_time[decode_time['event'] == 'omission'],
             ax=ax3, lw=2)
ax3.set(ylabel='Decoding performance (% correct)', xlabel='Time (s)', title='Reward omission',
        ylim=[40, 65])

sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine()
plt.tight_layout(pad=2)
plt.savefig(join(FIG_PATH, 'decoding_block_time_acc'))

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
ax1.plot([0, 0], ax1.get_ylim(), linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='f1', data=shuffle_time[decode_time['event'] == 'go cue'],
             ax=ax1, lw=2)
sns.lineplot(x='win_center', y='f1', data=decode_time[decode_time['event'] == 'go cue'],
             ax=ax1, lw=2)
ax1.set(ylabel='Decoding performance (F1 score)', xlabel='Time (s)', title='Stimulus onset')


ax2.plot([0, 0], ax1.get_ylim(), linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='f1', data=shuffle_time[decode_time['event'] == 'reward'],
             ax=ax2, lw=2)
sns.lineplot(x='win_center', y='f1', data=decode_time[decode_time['event'] == 'reward'],
             ax=ax2, lw=2)
ax2.set(ylabel='Decoding performance (%)', xlabel='Time (s)', title='Reward delivery')

ax3.plot([0, 0], ax1.get_ylim(), linestyle='dashed', color=[0.6, 0.6, 0.6])
sns.lineplot(x='win_center', y='f1', data=shuffle_time[decode_time['event'] == 'omission'],
             ax=ax3, lw=2)
sns.lineplot(x='win_center', y='f1', data=decode_time[decode_time['event'] == 'omission'],
             ax=ax3)
ax3.set(ylabel='Decoding performance (%)', xlabel='Time (s)', title='Reward omission')

sns.set(context='paper', font_scale=1.5, style='ticks')
sns.despine()
plt.savefig(join(FIG_PATH, 'decoding_block_time_f1'))
"""