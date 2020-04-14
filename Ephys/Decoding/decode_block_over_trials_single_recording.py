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

# Session
LAB = 'danlab'
SUBJECT = 'DY_011'
DATE = '2020-01-30'
PROBE = '00'

# Settings
DECODER = 'bayes'
N_NEURONS = 160
PRE_TIME = 0.6
POST_TIME = -0.2
NUM_SPLITS = 5
ITERATIONS = 500
TRIAL_WIN = [5, 15]

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

# Decode block probability
decode_result = decode(spikes.times, spikes.clusters, gocue_times, gocue_blocks,
                       pre_time=PRE_TIME, post_time=POST_TIME,
                       classifier='bayes', cross_validation='kfold',
                       n_neurons=N_NEURONS, iterations=ITERATIONS)

# Get decoding probability centered at block switches
prob_left = trials.probabilityLeft[(trials.probabilityLeft > 0.55)
                                   | (trials.probabilityLeft < 0.45)]
decode_prob = decode_result['probabilities'].mean(axis=0)
switch_to_l = [i for i, x in enumerate(np.diff(prob_left) > 0.3) if x]
switch_to_r = [i for i, x in enumerate(np.diff(prob_left) < -0.3) if x]
all_switches = np.append(switch_to_l, switch_to_r)
switch_sides = np.append(['left']*len(switch_to_l), ['right']*len(switch_to_r))
block_switch = pd.DataFrame(columns=['prob', 'trial_center', 'switch_side'])
for s, switch in enumerate(all_switches):
    block_switch = block_switch.append(pd.DataFrame(
                                {'prob': decode_prob[np.arange(switch - TRIAL_WIN[0] - 1,
                                                               switch + TRIAL_WIN[1])],
                                 'trials': np.arange(switch - TRIAL_WIN[0] - 1,
                                                     switch + TRIAL_WIN[1]) - switch + 1,
                                 'switch': switch_sides[s]}), sort=False)

# %% Plot


f, ax1 = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks", context="paper", font_scale=2)
ax1.plot(np.arange(1, gocue_times.shape[0]+1), decode_prob, lw=2)
ax1.set_ylabel('Probability of left block')
ax1.set(ylim=[0, 1], xlabel='Trials')
ax12 = ax1.twinx()
ax12.plot(np.arange(1, gocue_times.shape[0]+1), prob_left, color='red', lw=2)
ax12.set_ylabel('Probability of left trial', color='tab:red')
ax12.tick_params(axis='y', colors='red')
ax12.set(xlabel='Trials', ylim=[0, 1])

plt.tight_layout()

f, ax1 = plt.subplots(1, 1, figsize=(12, 8))
sns.set(style="ticks", context="paper", font_scale=2)
ax1.plot(np.arange(1, gocue_times.shape[0]+1), decode_prob, lw=2)
ax1.set_ylabel('Probability of left block')
ax1.set(ylim=[0, 1], xlabel='Trials')
ax12 = ax1.twinx()
ax12.plot(np.arange(1, gocue_times.shape[0]+1), prob_left, color='red', lw=2)
ax12.set_ylabel('Probability of left trial', color='tab:red')
ax12.tick_params(axis='y', colors='red')
ax12.set(xlabel='Trials', ylim=[0, 1])

plt.tight_layout()

f, ax1 = plt.subplots(1, 1, figsize=(5, 5))
sns.lineplot(x='trials', y='prob', data=block_switch, hue='switch', ci=68, ax=ax1)
ax1.plot([0, 0], [0, 1], linestyle='dashed', color=[0.6, 0.6, 0.6])
ax1.legend()



