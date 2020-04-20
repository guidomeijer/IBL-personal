#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:28:36 2020
@author: guido
"""

from os.path import join
import matplotlib.pyplot as plt
from pathlib import Path
import alf.io
import pandas as pd
import brainbox as bb
import seaborn as sns
import numpy as np
from ephys_functions import paths, figure_style
from oneibl.one import ONE
one = ONE()

# Settings
MIN_CONTRAST = 0.1
ALPHA = 0.05


def one_session_path(eid):
    ses = one.alyx.rest('sessions', 'read', id=eid)
    return Path(one._par.CACHE_DIR, ses['lab'], 'Subjects', ses['subject'],
                ses['start_time'][:10], str(ses['number']).zfill(3))


# Get list of recordings
eids, ses_info = one.search(dataset_types='spikes.times',
                            task_protocol='_iblrig_tasks_ephysChoiceWorld', details=True)

# Set path to save plots
DATA_PATH, FIG_PATH, SAVE_PATH = paths()
FIG_PATH = join(FIG_PATH, 'WholeBrain')

resp = pd.DataFrame()
for i, eid in enumerate(eids):

    # Load in data
    print('Processing session %d of %d' % (i+1, len(eids)))
    session_path = one_session_path(eid)
    trials = one.load_object(eid, 'trials')
    probes = one.load_object(eid, 'probes', download_only=False)
    if ((not hasattr(trials, 'stimOn_times'))
            or (len(trials.feedback_times) != len(trials.feedbackType))
            or (len(trials.stimOn_times) != len(trials.probabilityLeft))
            or (not hasattr(probes, 'trajectory'))):
        continue
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
                                            pre_time=[0.3, 0], post_time=[0, 0.3], alpha=ALPHA)[0]
        sig_rew = bb.task.responsive_units(spikes.times, spikes.clusters,
                                           trials.feedback_times[trials.feedbackType == 1],
                                           pre_time=[0.4, 0.1], post_time=[0.1, 0.4],
                                           alpha=ALPHA)[0]
        sig_omit = bb.task.responsive_units(spikes.times, spikes.clusters,
                                            trials.feedback_times[trials.feedbackType == -1],
                                            pre_time=[0.4, 0.1], post_time=[0.1, 0.4],
                                            alpha=ALPHA)[0]

        # Get choice neurons
        event_times = trials.response_times[(trials.choice == -1) | (trials.choice == 1)]
        event_choices = (trials.choice[
                            (trials.choice == -1) | (trials.choice == 1)] == 1).astype(int)
        sig_choice = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                 event_times, event_choices,
                                                 pre_time=0.3, post_time=0,
                                                 alpha=ALPHA)[0]

        # Get visual side neurons
        event_times = trials.stimOn_times[(trials.contrastRight > MIN_CONTRAST)
                                          | (trials.contrastLeft > MIN_CONTRAST)]
        event_sides = np.isnan(trials.contrastRight[
                            (trials.contrastRight > MIN_CONTRAST)
                            | (trials.contrastLeft > MIN_CONTRAST)]).astype(int)
        sig_side = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                               event_times, event_sides,
                                               pre_time=0, post_time=0.3,
                                               alpha=ALPHA)[0]

        # Get block neurons
        incl_trials = (trials.probabilityLeft > 0.55) | (trials.probabilityLeft < 0.45)
        trial_times = trials.goCue_times[incl_trials]
        trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)
        sig_block = bb.task.differentiate_units(spikes.times, spikes.clusters,
                                                trial_times, trial_blocks,
                                                pre_time=0.5, post_time=-0.2, alpha=ALPHA)[0]

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
                                              'block': (sig_block.shape[0]
                                                        / len(np.unique(spikes.clusters))),
                                              'ML': probes.trajectory[p]['x'],
                                              'AP': probes.trajectory[p]['y'],
                                              'DV': probes.trajectory[p]['z'],
                                              'phi': probes.trajectory[p]['phi'],
                                              'theta': probes.trajectory[p]['theta'],
                                              'depth': probes.trajectory[p]['depth']}))


resp.to_csv(join(SAVE_PATH, 'all_responsive_units.csv'))

# %% Plots

resp = pd.read_csv(join(SAVE_PATH, 'all_responsive_units.csv'))

# Plot histograms
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
sns.set(style="ticks", context="paper", font_scale=2)
BINS = 30
sns.distplot(resp['stim'], kde=False, bins=BINS, label='Stimulus', ax=ax1)
sns.distplot(resp['reward'], kde=False, bins=BINS, label='Reward', ax=ax1)
sns.distplot(resp['omit'], kde=False, bins=BINS, label='Omission', ax=ax1)
sns.distplot(resp['choice'], kde=False, bins=20, label='Left vs Right', ax=ax1)
ax1.legend(frameon=False)
ax1.set(ylabel='Count', xlabel='Proportion of responsive neurons')

sns.distplot(resp['choice'], kde=False, bins=BINS, label='Choice', ax=ax2)
sns.distplot(resp['block'], kde=False, bins=BINS, label='Stimulus prior', ax=ax2)
ax2.legend(frameon=False)
ax2.set(ylabel='Count', xlabel='Proportion of responsive neurons', xlim=[0, 0.36])

figure_style()
plt.savefig(join(FIG_PATH, 'hist_all_resp_units.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, 'hist_all_resp_units.png'), dpi=300)

# Plot map of shallow recordings
shallow = resp[(resp['ML'] < 0) & (resp['phi'] == 180)]

Y_LIM = [-6000, 4000]
X_LIM = [-5000, 5000]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(18, 7))
"""
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
"""
ax1.scatter([0, 0], [-4200, 0], s=50, color='black')
ax1.text(250, 0, 'Bregma', va='center')
ax1.text(250, -4200, 'Lambda', va='center')
sns.scatterplot(x='ML', y='AP', data=shallow.sort_values(by='stim'), size='n_neurons', hue='stim',
                palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200), hue_norm=(0, 1), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Stimulus', facecolor=[0.8, 0.8, 0.8])
ax1.get_legend().remove()

"""
ax2.plot([0, 0], [-4200, 0], color='k')
ax2.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax2.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax2.plot([X_LIM[0], 0], [2000, 0], color='k')
ax2.plot([0, X_LIM[1]], [-0, 2000], color='k')
"""
ax2.scatter([0, 0], [-4200, 0], s=50, color='black')
ax2.text(250, 0, 'Bregma', va='center')
ax2.text(250, -4200, 'Lambda', va='center')
sns.scatterplot(x='ML', y='AP', data=shallow.sort_values(by='reward'), size='n_neurons',
                hue='reward', palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200),
                hue_norm=(0, 1), ax=ax2)
ax2.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Reward', facecolor=[0.8, 0.8, 0.8])
ax2.get_legend().remove()

"""
ax3.plot([0, 0], [-4200, 0], color='k')
ax3.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax3.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax3.plot([X_LIM[0], 0], [2000, 0], color='k')
ax3.plot([0, X_LIM[1]], [-0, 2000], color='k')
"""
# ax3.scatter([0, 0], [-4200, 0], s=50, color='black')
# ax3.text(250, 0, 'Bregma', va='center')
# ax3.text(250, -4200, 'Lambda', va='center')
plot_h = sns.scatterplot(x='ML', y='AP', data=shallow.sort_values(by='omit'),
                         size='n_neurons', hue='omit',
                         palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200), hue_norm=(0, 1),
                         ax=ax3)
ax3.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Reward omission', facecolor=[0.8, 0.8, 0.8])

# Fix legend
leg = plot_h.legend(loc=(0.57, 0.3), facecolor=[0.8, 0.8, 0.8], framealpha=1)
leg.texts[0].set_text('% cells')
leg.texts[1].set_text('25')
leg.texts[2].set_text('50')
leg.texts[3].set_text('75')
leg.texts[4].set_text('100')
leg.texts[5].set_text('# cells')

figure_style(despine=False)
plt.savefig(join(FIG_PATH, 'responsive_unit_map_shallow.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, 'responsive_unit_map_shallow.png'), dpi=300)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 7))
"""
ax1.plot([0, 0], [-4200, 0], color='k')
ax1.scatter(0, 0, color='k')
ax1.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax1.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax1.plot([X_LIM[0], 0], [2000, 0], color='k')
ax1.plot([0, X_LIM[1]], [-0, 2000], color='k')
"""
plot_h = sns.scatterplot(x='ML', y='AP', data=shallow.sort_values(by='choice'),
                         size='n_neurons', hue='choice',
                         palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200),
                         hue_norm=(0, 0.4), ax=ax1)
ax1.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Left vs. Right trials', facecolor=[0.8, 0.8, 0.8])

# Fix legend
leg = plot_h.legend(loc=(0.65, 0.3), facecolor=[0.8, 0.8, 0.8], framealpha=1)

leg.texts[0].set_text('% cells')
leg.texts[1].set_text('10')
leg.texts[2].set_text('20')
leg.texts[3].set_text('30')
leg.texts[4].set_text('40')
leg.texts[5].set_text('# cells')

"""
ax2.plot([0, 0], [-4200, 0], color='k')
ax2.plot([X_LIM[0], 0], [-6000, -4200], color='k')
ax2.plot([0, X_LIM[1]], [-4200, -6000], color='k')
ax2.plot([X_LIM[0], 0], [2000, 0], color='k')
ax2.plot([0, X_LIM[1]], [-0, 2000], color='k')
"""
plot_h = sns.scatterplot(x='ML', y='AP', data=shallow.sort_values(by='block'),
                         size='n_neurons', hue='block',
                         palette='YlOrRd', size_norm=(50, 600), sizes=(50, 200),
                         hue_norm=(0, 0.1), ax=ax2)
ax2.set(xlim=X_LIM, ylim=Y_LIM, ylabel='AP coordinates (um)',
        xlabel='ML coordinates (um)', title='Stimulus prior', facecolor=[0.8, 0.8, 0.8])

# Fix legend
leg = plot_h.legend(loc=(0.65, 0.3), facecolor=[0.8, 0.8, 0.8], framealpha=1)

leg.texts[0].set_text('% cells')
leg.texts[1].set_text('0')
leg.texts[2].set_text('4')
leg.texts[3].set_text('8')
leg.texts[4].set_text('12')
leg.texts[5].set_text('# cels')

figure_style(despine=False)
plt.savefig(join(FIG_PATH, 'choice_prior_responsive_unit_map_shallow.pdf'), dpi=300)
plt.savefig(join(FIG_PATH, 'choice_prior_responsive_unit_map_shallow.png'), dpi=300)
