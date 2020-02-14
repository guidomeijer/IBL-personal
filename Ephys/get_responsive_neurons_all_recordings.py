#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:28:36 2020

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
from ephys_functions import paths
from oneibl.one import ONE
one = ONE()

MIN_CONTRAST = 0.1


def one_session_path(eid):
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times', details=True)

# Set path to save plots
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

resp = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
    if ((not hasattr(trials, 'stimOn_times'))
            or (len(trials.feedback_times) != len(trials.feedbackType))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))):
        continue
    probes = one.load_object(eid, 'probes', download_only=False)
    for p in range(len(probes['trajectory'])):
        probe_path = session_path.joinpath('alf', probes['description'][p]['label'])
        try:
            spikes = alf.io.load_object(probe_path, object='spikes')
            clusters = alf.io.load_object(probe_path, object='clusters')
        except Exception:
            continue
        if not hasattr(spikes, 'times'):
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
        sig_stim = bb.task.responsive_units(spikes.times, spikes.clusters, trials.stimOn_times,
                                            pre_time=[0.3, 0], post_time=[0, 0.3], alpha=0.01)[0]
        sig_rew = bb.task.responsive_units(spikes.times, spikes.clusters,
                                           trials.feedback_times[trials.feedbackType == 1],
                                           pre_time=[0.4, 0.1], post_time=[0.1, 0.4],
                                           alpha=0.01)[0]
        sig_omit = bb.task.responsive_units(spikes.times, spikes.clusters,
                                            trials.feedback_times[trials.feedbackType == -1],
                                            pre_time=[0.4, 0.1], post_time=[0.1, 0.4],
                                            alpha=0.01)[0]

        # Get choice neurons
        event_times = trials.stimOn_times[(trials.choice == -1) | (trials.choice == 1)]
        event_choices = (trials.choice[
                            (trials.choice == -1) | (trials.choice == 1)] == 1).astype(int)
        sig_choice = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                 event_times, event_choices,
                                                 alpha=0.01)[0]

        # Get visual side neurons
        event_times = trials.stimOn_times[(trials.contrastRight > MIN_CONTRAST)
                                          | (trials.contrastLeft > MIN_CONTRAST)]
        event_sides = np.isnan(trials.contrastRight[
                            (trials.contrastRight > MIN_CONTRAST)
                            | (trials.contrastLeft > MIN_CONTRAST)]).astype(int)
        sig_side = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                               event_times, event_sides,
                                               alpha=0.01)[0]

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
                                              'side': (sig_side.shape[0]
                                                       / len(np.unique(spikes.clusters))),
                                              'ML': probes.trajectory[p]['x'],
                                              'AP': probes.trajectory[p]['y'],
                                              'DV': probes.trajectory[p]['z'],
                                              'phi': probes.trajectory[p]['phi'],
                                              'theta': probes.trajectory[p]['theta'],
                                              'depth': probes.trajectory[p]['depth']}))

resp.to_csv(join(SAVE_PATH, 'responsive_units_map.csv'))

# %% Plot

Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18, 8))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='stim',
                palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200), hue_norm=(0, 1), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Stimulus')
ax1.get_legend().remove()

ax2.plot([0, 0], [-4200, 0], color='k')
ax2.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax2.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax2.plot([X_LIM[0], 0], [2000, 0], color='k')
ax2.plot([0, X_LIM[1]], [-0, 2000], color='k')
sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='reward',
                palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200), hue_norm=(0, 1), ax=ax2)
ax2.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Reward')
ax2.get_legend().remove()

ax3.plot([0, 0], [-4200, 0], color='k')
ax3.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax3.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax3.plot([X_LIM[0], 0], [2000, 0], color='k')
ax3.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='omit',
                         palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200), hue_norm=(0, 1),
                         ax=ax3)
ax3.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Reward omission')

# Fix legend
leg = plot_h.legend(loc=(1.05, 0.5))
leg.texts[0].set_text('Prop. neurons')
leg.texts[1].set_text('0.25')
leg.texts[2].set_text('0.5')
leg.texts[3].set_text('0.75')
leg.texts[4].set_text('1')
leg.texts[5].set_text('# neurons')

plt.savefig(join(FIG_PATH, 'all_responsive_unit_map'))


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 7))
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.scatter(0, 0, color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='choice',
                         palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200),
                         hue_norm=(0, 0.45), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Choice')
ax1.get_legend().remove()

ax2.plot([0, 0], [-4200, 0], color='k')
ax2.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax2.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax2.plot([X_LIM[0], 0], [2000, 0], color='k')
ax2.plot([0, X_LIM[1]], [-0, 2000], color='k')
plot_h = sns.scatterplot(x='ML', y='AP', data=resp, size='n_neurons', hue='side',
                         palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200),
                         hue_norm=(0, 0.45), ax=ax2)
ax2.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Stimulus side')

# Fix legend
leg = plot_h.legend(loc=(1.05, 0.5))
leg.texts[0].set_text('Prop. neurons')
leg.texts[1].set_text('0.05')
leg.texts[2].set_text('0.15')
leg.texts[3].set_text('0.30')
leg.texts[4].set_text('0.45')
leg.texts[5].set_text('# neurons')

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'choice_responsive_unit_map'))
