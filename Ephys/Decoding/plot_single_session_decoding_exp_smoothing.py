#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Guido Meijer
"""

from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import brainbox.io.one as bbone
from sklearn.model_selection import KFold
from brainbox.population.decode import get_spike_counts_in_bins, regress
from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prev_action
from models.expSmoothing_stimside import expSmoothing_stimside as exp_stimside
from scipy.stats import pearsonr
from my_functions import paths, figure_style, get_full_region_name, load_trials
from oneibl.one import ONE
one = ONE()

# Settings
SUBJECT = 'KS022'
EID = '15f742e1-1043-45c9-9504-f1e8a53c1744'
PROBE = 'probe01'
TARGET = 'prior-prevaction'
CHANCE_LEVEL = 'other-trials'
DECODER = 'linear-regression'
INCL_NEURONS = 'pass-QC'
INCL_SESSIONS = 'aligned-behavior'
VALIDATION = 'kfold'
ATLAS = 'beryl-atlas'
TIME_WIN = '600--100'
NUM_SPLITS = 5
PRE_TIME = 0.6
ITERATIONS = 50
POST_TIME = -0.1
YLIM = [-.4, .71]
DPI = 300
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'Ephys', 'Decoding')
FULL_NAME = True
SAVE_FIG = True
OVER_CHANCE = True

# %% Plot

# Load in data all clusters
decoding_result = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                                      f'{TARGET}_{CHANCE_LEVEL}_{VALIDATION}_{INCL_SESSIONS}_' \
                                      f'all_cells_{ATLAS}_{TIME_WIN}.p'))
decoding_result['metric_plot'] = decoding_result['r'] - decoding_result['r_null']
decoding_result['full_region'] = get_full_region_name(decoding_result['region'].values)

# Load in data pass QC clusters
decoding_pass = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', DECODER,
                                      f'{TARGET}_{CHANCE_LEVEL}_{VALIDATION}_{INCL_SESSIONS}_' \
                                      f'pass-QC_cells_{ATLAS}_{TIME_WIN}.p'))
decoding_pass['metric_plot'] = decoding_pass['r'] - decoding_pass['r_null']
decoding_pass['full_region'] = get_full_region_name(decoding_pass['region'].values)

# %% Decode with all clusters of the probe

all_trials = pd.read_pickle(join(SAVE_PATH, 'Ephys', 'Decoding', 'all_trials.p'))
all_eids = np.sort(decoding_result.loc[decoding_result['subject'] == SUBJECT, 'eid'].unique())

 # Load in fitted behavioral model
if TARGET == 'prior-prevaction':
    model = exp_prev_action(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                            all_eids, SUBJECT,
                            np.array([np.array(None)] * all_eids.shape[0]),
                            np.array([np.array(None)] * all_eids.shape[0]),
                            np.array([np.array(None)] * all_eids.shape[0]))
elif TARGET == 'prior-stimside':
    model = exp_stimside(join(SAVE_PATH, 'Behavior', 'exp_smoothing_model_fits/'),
                         all_eids, SUBJECT,
                         np.array([np.array(None)] * all_eids.shape[0]),
                         np.array([np.array(None)] * all_eids.shape[0]),
                         np.array([np.array(None)] * all_eids.shape[0]))
model.load_or_train(nb_steps=2000, remove_old=False)
params = model.get_parameters(parameter_type='posterior_mean')

# Get priors per trial
trials = load_trials(EID, invert_stimside=True)
priors = model.compute_signal(signal='prior', act=np.array(trials['choice']),
                              stim=np.array(trials['signed_contrast']),
                              side=np.array(trials['stim_side']),
                              parameter_type='posterior_mean')['prior']

# Get clusters in this brain region
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(EID, aligned=True, one=one)

# Get list of neurons that pass QC
if INCL_NEURONS == 'pass-QC':
    clusters_pass = np.where(clusters[PROBE]['metrics']['label'] == 1)[0]
elif INCL_NEURONS == 'all':
    clusters_pass = np.arange(clusters[PROBE]['metrics'].shape[0])

# Select spikes and clusters
spike_times = spikes[PROBE].times[np.isin(spikes[PROBE].clusters, clusters_pass)]
spike_clusters = spikes[PROBE].clusters[np.isin(spikes[PROBE].clusters, clusters_pass)]

# Decode prior from model fit
times = np.column_stack(((trials.goCue_times - PRE_TIME), (trials.goCue_times + POST_TIME)))
pop_act_all, cluster_ids = get_spike_counts_in_bins(spikes[PROBE].times, spikes[PROBE].clusters, times)
pop_act_all = pop_act_all.T
pop_act_pass, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)
pop_act_pass = pop_act_pass.T
if VALIDATION == 'kfold-interleaved':
    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
elif VALIDATION == 'kfold':
    cv = KFold(n_splits=NUM_SPLITS, shuffle=False)
pred_all = regress(pop_act_all, priors, cross_validation=cv, regularization='L1')
r_all = pearsonr(priors, pred_all)[0]
pred_pass = regress(pop_act_pass, priors, cross_validation=cv, regularization='L1')
r_pass = pearsonr(priors, pred_pass)[0]

# Estimate chance level
r_null_all, r_null_pass = np.empty(ITERATIONS), np.empty(ITERATIONS)
for k in range(ITERATIONS):
    # Exclude the current mice from all trials
    all_trials_excl = all_trials[all_trials['subject'] != SUBJECT]

    # Get a random chunck of trials the same length as the current session
    null_selection = np.random.randint(all_trials_excl.shape[0])
    while null_selection + trials.shape[0] >= all_trials_excl.shape[0]:
        null_selection = np.random.randint(all_trials_excl.shape[0])
    null_trials = all_trials_excl[null_selection : (null_selection
                                                    + trials.shape[0])]

    # Get null target
    if 'prior' in TARGET:
        signal = 'prior'
    elif 'prederr' in TARGET:
        signal = 'prediction_error'
    null_target = model.compute_signal(signal=signal,
                                       act=null_trials['choice'].values,
                                       stim=null_trials['signed_contrast'].values,
                                       side=null_trials['stim_side'].values,
                                       parameter_type='posterior_mean',
                                       verbose=False)[signal]
    null_target = np.squeeze(np.array(null_target))

    if 'abs' in TARGET:
        null_target = np.abs(null_target)

    # Decode prior of null trials
    null_pred, null_pred_train = regress(pop_act_all, null_target, regularization='L1',
                                         cross_validation=cv, return_training=True)
    r_null_all[k] = pearsonr(null_target, null_pred)[0]
    null_pred, null_pred_train = regress(pop_act_pass, null_target, regularization='L1',
                                         cross_validation=cv, return_training=True)
    r_null_pass[k] = pearsonr(null_target, null_pred)[0]

r_over_null_all = r_all - r_null_all.mean()
p_all = np.sum(r_all < r_null_all) / len(r_null_all)
r_over_null_pass = r_pass - r_null_pass.mean()
p_pass = np.sum(r_pass < r_null_pass) / len(r_null_pass)

# %% Plot

# Restructure dataframe for plotting
plot_decoding = decoding_result[decoding_result['region'].isin(
                    decoding_result['region'][decoding_result['eid'] == EID])].copy()
plot_decoding['clusters'] = 'all'
plot_decoding['color'] = 0
plot_decoding.loc[plot_decoding['eid'] == EID, 'color'] = 1

pass_plot = decoding_pass.copy()
pass_plot['clusters'] = 'pass'
pass_plot['color'] = 0
pass_plot.loc[decoding_pass['eid'] == EID, 'color'] = 2
plot_decoding = plot_decoding.append(pass_plot[pass_plot['region'].isin(
                    pass_plot['region'][pass_plot['eid'] == EID])]).copy()
plot_decoding = plot_decoding.reset_index(drop=True)

for i in plot_decoding.index.values:
    if plot_decoding.loc[i, 'clusters'] == 'all':
        plot_decoding.loc[i, 'example_metric'] =  plot_decoding.loc[(
            (plot_decoding['region'] == plot_decoding.loc[i, 'region'])
            & (plot_decoding['eid'] == EID)
            & (plot_decoding['clusters'] == 'all')), 'metric_plot'].values
    elif plot_decoding.loc[i, 'clusters'] == 'pass':
        plot_decoding.loc[i, 'example_metric'] =  plot_decoding.loc[(
            (plot_decoding['region'] == plot_decoding.loc[i, 'region'])
            & (plot_decoding['eid'] == EID)
            & (plot_decoding['clusters'] == 'pass')), 'metric_plot'].values

plot_decoding.loc[plot_decoding.shape[0] + 1, 'full_region'] = 'All regions'
plot_decoding.loc[plot_decoding.shape[0], 'example_metric'] = 1
plot_decoding.loc[plot_decoding.shape[0], 'metric_plot'] = r_over_null_all
plot_decoding.loc[plot_decoding.shape[0], 'color'] = 1
plot_decoding.loc[plot_decoding.shape[0] + 1, 'full_region'] = 'All regions'
plot_decoding.loc[plot_decoding.shape[0], 'example_metric'] = 1
plot_decoding.loc[plot_decoding.shape[0], 'metric_plot'] = r_over_null_pass
plot_decoding.loc[plot_decoding.shape[0], 'color'] = 2

plot_decoding = plot_decoding.sort_values('example_metric', ascending=False)

colors = [(.8, .8, .8), (1, 0, 0), (0, 0, 1)]

figure_style(font_scale=2.5)
f, ax1 = plt.subplots(figsize=(18, 10))
if 'prior-prevaction' in TARGET:
    target_str = 'prior (previous actions)'
elif 'prior-stimside' in TARGET:
    target_str = 'prior (stimulus sides)'
elif 'prederr-pos' in TARGET:
    target_str = 'positive prediction error'
elif 'prederr-neg' in TARGET:
    target_str = 'negative prediction error'
elif 'prior-stim' in TARGET:
    target_str = 'prior during 0% contrast trials'
elif 'prior-norm' in TARGET:
    target_str = 'prior during stimulus (normalized firing rates)'
elif 'prederr-abs' in TARGET:
    target_str = 'unsigned prediction error'
if VALIDATION == 'kfold':
    val_str = 'continuous 5-fold'
elif VALIDATION == 'kfold-interleaved':
    val_str = 'interleaved 5-fold'
f.suptitle('Decoding of %s using linear regression with %s cross-validation' % (
    target_str, val_str), fontsize=20)
"""
ax_lines = sns.pointplot(x='metric_plot', y='full_region', data=session_decoding,
                         join=False, color='red',
                         markers="|", scale=2.5, ax=ax1)
plt.setp(ax_lines.collections, zorder=100, label="")
"""
#sns.stripplot(x='metric_plot', y='full_region', data=plot_decoding, s=8,
#              hue='example_session', palette=colors, dodge=True, ax=ax1)
plot_handle = sns.swarmplot(x='metric_plot', y='full_region', data=plot_decoding, s=8,
                            hue='color', palette=colors, ax=ax1)
legend = ax1.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
legend.texts[0].set_text('Other recordings')
legend.texts[1].set_text('All neurons')
legend.texts[2].set_text('QC neurons')

#ax1.legend_.remove()
#sns.stripplot(x='metric_plot', y='full_region', data=other_decoding, s=6,
#              hue='p_value', palette='Greys', ax=ax1)
#sns.violinplot(x='r_prior_plot', y='full_region', data=other_decoding, ax=ax1)
ax1.plot([0, 0], ax1.get_ylim(), color=[0.5, 0.5, 0.5], ls='--')
if OVER_CHANCE:
    # str_xlabel = 'Decoding improvement over pseudo sessions (mean squared error)'
    str_xlabel = 'Decoding improvement over null ($R^2$)'
else:
    str_xlabel = 'Decoding performance (r)'
ax1.set(xlabel=str_xlabel, ylabel='', xlim=YLIM)
plt.tight_layout()
sns.despine(trim=True, offset=0)

