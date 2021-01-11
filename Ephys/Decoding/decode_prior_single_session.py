#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 13:28:36 2020

@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from brainbox.task import generate_pseudo_stimuli, generate_pseudo_session
from brainbox.population import linear_regression
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr
import seaborn as sns
from prior_funcs import perform_inference
import alf
from ephys_functions import paths, combine_layers_cortex, figure_style
import brainbox.io.one as bbone
from oneibl.one import ONE
one = ONE()

"""
EID = '15f742e1-1043-45c9-9504-f1e8a53c1744'
REGION = 'SNr'
PROBE = 'probe01'
"""

EID = 'b658bc7d-07cd-4203-8a25-7b16b549851b'
REGION = 'CP'
PROBE = 'probe00'

PRE_TIME = 0.6
POST_TIME = -0.1
NUM_SPLITS = 5
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
# FIG_PATH = join(FIG_PATH, 'Decoding', 'Sessions', DECODER)

# %%
# Load in data
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(EID, aligned=True, one=one)
ses_path = one.path_from_eid(EID)
trials = alf.io.load_object(join(ses_path, 'alf'), 'trials')

# Get trial vectors
_, trials.contrastLeft, trials.contrastRight = generate_pseudo_stimuli(trials.probabilityLeft.shape[0])

stim_side = (np.array(np.isnan(trials.contrastLeft)==False) * -1
             + np.array(np.isnan(trials.contrastRight)==False)) * 1
infer_p_left = perform_inference(stim_side)[0]
infer_p_left = infer_p_left[:, 0]

# Get clusters in this brain region
region_clusters = combine_layers_cortex(clusters[PROBE]['acronym'])
clusters_in_region = clusters[PROBE].metrics.cluster_id[region_clusters == REGION]

# Select spikes and clusters
spks_region = spikes[PROBE].times[np.isin(spikes[PROBE].clusters, clusters_in_region)]
clus_region = spikes[PROBE].clusters[np.isin(spikes[PROBE].clusters,
                                             clusters_in_region)]

# Decode
pred_none = linear_regression(spks_region, clus_region, trials['stimOn_times'], infer_p_left,
                              pre_time=PRE_TIME, post_time=POST_TIME,
                              cross_validation='none')
pred_kfold_int = linear_regression(spks_region, clus_region, trials['stimOn_times'], infer_p_left,
                                   pre_time=PRE_TIME, post_time=POST_TIME,
                                  cross_validation='kfold-interleaved', num_splits=NUM_SPLITS)
pred_kfold_cont = linear_regression(spks_region, clus_region, trials['stimOn_times'], infer_p_left,
                                    pre_time=PRE_TIME, post_time=POST_TIME,
                                    cross_validation='kfold', num_splits=NUM_SPLITS)

pearsonr(pred_kfold_int, infer_p_left)
pred_blocks = linear_regression(spks_region, clus_region, trials['stimOn_times'],
                                trials.probabilityLeft,
                                pre_time=PRE_TIME, post_time=POST_TIME,
                                cross_validation='kfold-interleaved', num_splits=NUM_SPLITS)
pearsonr(pred_blocks, trials.probabilityLeft)

# %%

pred_kfold_int[pred_kfold_int > 2] = np.nan
pred_kfold_int[pred_kfold_int < -2] = np.nan
pred_kfold_cont[pred_kfold_cont > 2] = np.nan
pred_kfold_cont[pred_kfold_cont < -2] = np.nan


figure_style(font_scale=4)
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 20), sharex=True)
ax1.plot(np.arange(1, trials['probabilityLeft'].shape[0] + 1), trials['probabilityLeft'],
         label='pLeft')
ax1.plot(np.arange(1, trials['probabilityLeft'].shape[0] + 1), infer_p_left,
         label='inferred pLeft')
ax1.plot(np.arange(1, trials['probabilityLeft'].shape[0] + 1), pred_kfold_int,
         label='predicted pLeft')
ax1.legend(frameon=False)
ax1.set(ylabel='Probability of left stimulus',
        title='Linear regression prediction (5-fold interleaved), region: %s' % REGION)

ax2.plot(np.arange(1, trials['probabilityLeft'].shape[0] + 1), pred_none,
         label='No cross-validation')
ax2.plot(np.arange(1, trials['probabilityLeft'].shape[0] + 1), pred_kfold_int,
         label='5-fold interleaved')
ax2.plot(np.arange(1, trials['probabilityLeft'].shape[0] + 1), pred_kfold_cont,
         label='5-fold continuous')
ax2.legend(frameon=False)
ax2.set(ylabel='Probability of left stimulus', xlabel='Trials',
        title='Cross-validation comparison')

plt.tight_layout(pad=4)
sns.despine()
# plt.savefig(join(FIG_PATH, '%s_%s' % (REGION, DECODER)))
