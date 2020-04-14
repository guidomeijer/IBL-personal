#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:10:00 2020

@author: guido
"""

from os import listdir
from os.path import join
import alf.io as ioalf
import numpy as np
from brainbox.population import decode, lda_project
from ephys_functions import paths

# Session
LAB = 'mainenlab'
SUBJECT = 'ZM_2240'
DATE = '2020-01-23'
PROBE = '00'

# Settings
PRE_TIME = 0.5
POST_TIME = 0

DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Decoding', 'ContraStim')

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

# Get trial vectors
incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
trial_times = trials.goCue_times[incl_trials]
probability_left = trials.probabilityLeft[incl_trials]
trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

# Decode block identity
bayes_kfold = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier='bayes', cross_validation='kfold')
bayes_block = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                     pre_time=PRE_TIME, post_time=POST_TIME,
                     classifier='bayes', cross_validation='block', prob_left=probability_left)
bayes_loo = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                   pre_time=PRE_TIME, post_time=POST_TIME,
                   classifier='bayes', cross_validation='leave-one-out')
forest_kfold = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME,
                      classifier='forest', cross_validation='kfold')
forest_block = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                      pre_time=PRE_TIME, post_time=POST_TIME,
                      classifier='forest', cross_validation='block', prob_left=probability_left)
forest_loo = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                    pre_time=PRE_TIME, post_time=POST_TIME,
                    classifier='forest', cross_validation='leave-one-out')
lda_kfold = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                   pre_time=PRE_TIME, post_time=POST_TIME,
                   classifier='lda', cross_validation='kfold')
lda_block = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                   pre_time=PRE_TIME, post_time=POST_TIME,
                   classifier='lda', cross_validation='block', prob_left=probability_left)
lda_loo = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                 pre_time=PRE_TIME, post_time=POST_TIME,
                 classifier='lda', cross_validation='leave-one-out')

# LDA projection
lda_proj_kfold = lda_project(spikes.times, spikes.clusters, trial_times, trial_blocks,
                             pre_time=PRE_TIME, post_time=POST_TIME, cross_validation='kfold')
lda_proj_kfold = lda_project(spikes.times, spikes.clusters, trial_times, trial_blocks,
                             pre_time=PRE_TIME, post_time=POST_TIME, cross_validation='block',
                             prob_left=probability_left)
lda_proj_loo = lda_project(spikes.times, spikes.clusters, trial_times, trial_blocks,
                           pre_time=PRE_TIME, post_time=POST_TIME,
                           cross_validation='leave-one-out')


