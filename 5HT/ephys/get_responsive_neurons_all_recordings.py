#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:28:36 2020

Get all responsive neurons for all recordings

@author: guido
"""

from os import mkdir
from os.path import join, isdir
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import alf.io
import pandas as pd
import brainbox as bb
import seaborn as sns
import shutil
from scipy import stats
import numpy as np
from functions_5HT import paths, one_session_path
from oneibl.one import ONE
one = ONE()

# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times', details=True)

# Set path to save plots
DATA_PATH, FIG_PATH, _ = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

resp = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    session_path = one_session_path(eid)
    spikes = one.load_object(eid, 'spikes', download_only=True)
    trials = one.load_object(eid, 'trials')
    probes = one.load_object(eid, 'probes', download_only=False)
    if (not hasattr(trials, 'stimOn_times')
            or (trials.stimOn_times.shape[0] != trials.probabilityLeft.shape[0])
            or (not hasattr(probes, 'trajectory'))):
        continue
    for p in range(len(probes['trajectory'])):
        # Select shallow penetrations
        if (probes['trajectory'][p]['theta'] == 15) and (probes['trajectory'][p]['depth'] < 4500):
            probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
            try:
                spikes = alf.io.load_object(probe_path, object='spikes')
                clusters = alf.io.load_object(probe_path, object='clusters')
            except Exception:
                continue

            # Only use single units
            spikes.times = spikes.times[np.isin(
                    spikes.clusters, clusters.metrics.cluster_id[
                        clusters.metrics.ks2_label == 'good'])]
            spikes.clusters = spikes.clusters[np.isin(
                    spikes.clusters, clusters.metrics.cluster_id[
                        clusters.metrics.ks2_label == 'good'])]

            # Get session info
            nickname = ses_info[i]['subject']
            ses_date = ses_info[i]['start_time'][:10]

            # Get number of responsive neurons
            sig_stim = bb.task.responsive_units(spikes.times, spikes.clusters,
                                                trials.stimOn_times, alpha=0.01)[0]
            sig_rew = bb.task.responsive_units(spikes.times, spikes.clusters,
                                               trials.feedback_times[trials.feedbackType == 1],
                                               alpha=0.01)[0]
            sig_omit = bb.task.responsive_units(spikes.times, spikes.clusters,
                                                trials.feedback_times[trials.feedbackType == -1],
                                                alpha=0.01)[0]

            # Get choice neurons
            event_times = trials.stimOn_times[(trials.choice == -1) | (trials.choice == 1)]
            event_choices = (trials.choice[
                                (trials.choice == -1) | (trials.choice == 1)] == 1).astype(int)
            sig_choice = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                     event_times, event_choices,
                                                     alpha=0.01)[0]
            # Calculate whether neuron discriminates blocks
            trial_times = trials.goCue_times[
                                ((trials.probabilityLeft > 0.55)
                                 | (trials.probabilityLeft < 0.45))]
            trial_blocks = (trials.probabilityLeft[
                    (((trials.probabilityLeft > 0.55)
                      | (trials.probabilityLeft < 0.45)))] > 0.55).astype(int)

            sig_blocks = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                     trial_times, trial_blocks,
                                                     pre_time=1, post_time=0, alpha=0.01)[0]

            resp = resp.append(pd.DataFrame(index=[0],
                                            data={'subject': nickname,
                                                  'date': ses_date,
                                                  'eid': eid,
                                                  'n_neurons': len(np.unique(spikes.clusters)),
                                                  'stim': (sig_stim.shape[0]
                                                           / len(np.unique(spikes.clusters))),
                                                  'reward': (sig_rew.shape[0]
                                                             / len(np.unique(spikes.clusters))),
                                                  'omit': (sig_omit.shape[0]
                                                           / len(np.unique(spikes.clusters))),
                                                  'choice': (sig_choice.shape[0]
                                                             / len(np.unique(spikes.clusters))),
                                                  'blocks': (sig_blocks.shape[0]
                                                             / len(np.unique(spikes.clusters))),
                                                  'ML': probes.trajectory[p]['x'],
                                                  'AP': probes.trajectory[p]['y'],
                                                  'DV': probes.trajectory[p]['z'],
                                                  'phi': probes.trajectory[p]['phi'],
                                                  'theta': probes.trajectory[p]['theta'],
                                                  'depth': probes.trajectory[p]['depth']}))

# %% Plot

Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize=(18, 8))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='stim',
                palette='YlOrRd', sizes=(50, 200), hue_norm=(0, 1), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Stimulus')
ax1.get_legend().remove()

ax2.plot([0, 0], [-4200, 0], color='k')
ax2.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax2.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax2.plot([X_LIM[0], 0], [2000, 0], color='k')
ax2.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='reward',
                palette='YlOrRd', sizes=(50, 200), hue_norm=(0, 1), ax=ax2)
ax2.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Reward')
ax2.get_legend().remove()

ax3.plot([0, 0], [-4200, 0], color='k')
ax3.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax3.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax3.plot([X_LIM[0], 0], [2000, 0], color='k')
ax3.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='omit',
                palette='YlOrRd', sizes=(50, 200), hue_norm=(0, 1), ax=ax3)
ax3.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Reward omission')
ax3.get_legend().remove()

ax4.plot([0, 0], [-4200, 0], color='k')
ax4.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax4.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax4.plot([X_LIM[0], 0], [2000, 0], color='k')
ax4.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='choice',
                         palette='YlOrRd', sizes=(50, 200), hue_norm=(0, 1), ax=ax4)
ax4.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Choice')
ax4.legend(loc=(1.05, 0.5))
plt.savefig(join(FIG_PATH, 'responsive_unit_map'))

fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='choice',
                palette='YlOrRd', sizes=(50, 200), hue_norm=(0, 0.2), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Choice')
ax1.legend(loc=(0.9, 0.5))
plt.savefig(join(FIG_PATH, 'responsive_unit_map_choice'))

fig, ax1 = plt.subplots(1, 1, figsize=(6, 6))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='blocks',
                palette='YlOrRd', sizes=(50, 200), hue_norm=(0, 0.15), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Block identity')
ax1.legend(loc=(0.9, 0.5))

plt.savefig(join(FIG_PATH, 'responsive_unit_map_blocks'))
